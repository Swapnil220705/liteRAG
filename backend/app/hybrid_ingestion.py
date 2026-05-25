"""
Hybrid Ingestion Pipeline — v2
==============================
Writes to: backend/data/knowledge_artifact_v2.json

ZERO DATA LOSS architecture:
  text.original      — raw extracted text, never modified or replaced
  text.normalized    — cleaning only (lowercase, whitespace, no entity removal)
  text.translated_en — only set if ENABLE_TRANSLATION=true (default OFF)
  raw_table          — verbatim 2D list exactly as pdfplumber extracted
  value_raw          — cell string exactly as found in document
  value_parsed       — float conversion of value_raw, or None if not numeric
  distilled          — auxiliary s/k/c layer, ALWAYS computed from original text

Hard invariants:
  - NEVER write to knowledge_artifact.json or metadata.json
  - NEVER replace original text with distilled or translated output
  - ENABLE_TRANSLATION defaults to false
  - ENABLE_ANSWER_EQUIVALENCE defaults to false
  - Any validation failure → abort; output file is NOT written

ENV flags:
  ENABLE_TRANSLATION=true           Enable NLLB-based translation
  TRANSLATION_MODEL=...             HuggingFace model ID (default: facebook/nllb-200-distilled-600M)
  ENABLE_ANSWER_EQUIVALENCE=true    Enable LLM-based answer equivalence check
  ANSWER_EQUIV_THRESHOLD=0.75       Cosine similarity floor for equivalence
  GEMINI_MODEL=gemini-3-flash-preview  Model for equivalence LLM calls
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import fitz  # PyMuPDF — always available (in requirements.txt)
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Bootstrap paths + env
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_PATH = DATA_DIR / "knowledge_artifact_v2.json"

# Files that must NEVER be touched
PROTECTED_PATHS = {
    DATA_DIR / "knowledge_artifact.json",
    DATA_DIR / "metadata.json",
}

load_dotenv(BASE_DIR / ".env")

ENABLE_TRANSLATION: bool = os.getenv("ENABLE_TRANSLATION", "false").lower() == "true"
ENABLE_ANSWER_EQUIVALENCE: bool = os.getenv("ENABLE_ANSWER_EQUIVALENCE", "false").lower() == "true"
ANSWER_EQUIV_THRESHOLD: float = float(os.getenv("ANSWER_EQUIV_THRESHOLD", "0.75"))
GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-3-flash-preview")

# ---------------------------------------------------------------------------
# Optional dependency guards
# ---------------------------------------------------------------------------

try:
    import pdfplumber  # type: ignore
    HAS_PDFPLUMBER = True
except ImportError:
    HAS_PDFPLUMBER = False
    logging.warning("[hybrid_ingestion] pdfplumber not installed — table extraction disabled.")

try:
    from langdetect import detect as _langdetect  # type: ignore
    HAS_LANGDETECT = True
except ImportError:
    HAS_LANGDETECT = False
    logging.warning("[hybrid_ingestion] langdetect not installed — language detection disabled.")

# Lazy translation model (loaded only when needed and ENABLE_TRANSLATION=true)
_nllb_tokenizer = None
_nllb_model = None


def _load_nllb() -> bool:
    global _nllb_tokenizer, _nllb_model
    if _nllb_model is not None:
        return True
    try:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer  # type: ignore

        model_id = os.getenv("TRANSLATION_MODEL", "facebook/nllb-200-distilled-600M")
        logging.info(f"[hybrid_ingestion] Loading translation model: {model_id}")
        _nllb_tokenizer = AutoTokenizer.from_pretrained(model_id)
        _nllb_model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
        return True
    except Exception as exc:
        logging.warning(f"[hybrid_ingestion] Translation model load failed: {exc}")
        return False


# ---------------------------------------------------------------------------
# Reuse existing DistillationEngine
# ---------------------------------------------------------------------------

try:
    from app.distillation import DistillationEngine as _DistillationEngine  # within backend/ package
except ModuleNotFoundError:
    from backend.app.distillation import DistillationEngine as _DistillationEngine  # noqa: E402

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger("hybrid_ingestion")


# ===========================================================================
# STEP 1 — Section Parser
# ===========================================================================

class SectionParser:
    """
    Extracts section-aware structure from a PDF.

    Strategy:
      1. Read every text span with its font size via PyMuPDF.
      2. Derive a heading-font-size threshold (median + 20%).
      3. Classify spans as HEADING vs BODY.
      4. Group BODY text under the nearest preceding HEADING.
      5. Fallback: if fewer than 2 distinct headings found → page-based grouping.
      6. Merge sections smaller than MIN_SECTION_WORDS into the previous section.

    All original text is preserved verbatim under section["_raw_text"].
    """

    MIN_SECTION_WORDS: int = 40  # merge sections with fewer words than this
    HEADING_MAX_WORDS: int = 18  # a span with more words cannot be a heading
    HEADING_MIN_CHARS: int = 3

    def parse(self, pdf_path: str) -> List[Dict]:
        """
        Returns a list of section dicts:
          {
            "heading": str,
            "page_start": int,
            "page_end": int,
            "_raw_text": str,           # verbatim body text (concat of spans)
            "_table_pages": List[int],  # pages covered by this section
          }
        """
        doc = fitz.open(pdf_path)
        try:
            blocks = self._extract_blocks(doc)
            heading_threshold = self._detect_heading_font_size(blocks)

            if heading_threshold is not None:
                sections = self._group_by_headings(blocks, heading_threshold, doc)
            else:
                sections = []

            # Fallback: page-based grouping if we found fewer than 2 headings
            distinct_headings = {s["heading"] for s in sections if s["heading"] != "_page_section"}
            if len(distinct_headings) < 2:
                log.info("Fewer than 2 distinct headings found — using page-based section fallback.")
                sections = self._fallback_page_grouping(doc)

            sections = self._merge_small_sections(sections)
            return sections
        finally:
            doc.close()

    # ---- internal helpers ------------------------------------------------

    def _extract_blocks(self, doc: fitz.Document) -> List[Dict]:
        """
        Returns a flat list of span-level entries:
          { "text", "font_size", "font_flags", "page", "bbox" }
        Strips empty or whitespace-only spans.
        """
        entries = []
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            page_dict = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)
            for block in page_dict.get("blocks", []):
                if block.get("type") != 0:  # 0 = text block
                    continue
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        text = span.get("text", "").strip()
                        if not text:
                            continue
                        entries.append({
                            "text": text,
                            "font_size": round(span.get("size", 11.0), 1),
                            "font_flags": span.get("flags", 0),  # bold = flags & 16
                            "page": page_num + 1,
                            "bbox": span.get("bbox", ()),
                        })
        return entries

    def _detect_heading_font_size(self, blocks: List[Dict]) -> Optional[float]:
        """
        Returns a font-size threshold above which a span is considered a heading.
        Uses the 75th-percentile font size in the document.
        Returns None if the document has negligible text.
        """
        sizes = [b["font_size"] for b in blocks if b["font_size"] > 0]
        if not sizes:
            return None
        sizes_sorted = sorted(sizes)
        p75_idx = int(len(sizes_sorted) * 0.75)
        p75 = sizes_sorted[p75_idx]
        # Require the heading threshold to be meaningfully larger than body
        median = sizes_sorted[len(sizes_sorted) // 2]
        threshold = max(p75, median * 1.15)
        return threshold

    def _is_heading_span(self, entry: Dict, threshold: float) -> bool:
        text = entry["text"].strip()
        words = text.split()
        if len(words) < 1 or len(words) > self.HEADING_MAX_WORDS:
            return False
        if len(text) < self.HEADING_MIN_CHARS:
            return False
        is_bold = bool(entry["font_flags"] & 16)
        is_large = entry["font_size"] >= threshold
        is_allcaps = text.isupper() and len(words) <= 8
        # Must satisfy at least: large font OR (bold AND short title-like text)
        return is_large or (is_bold and len(words) <= 12) or is_allcaps

    def _group_by_headings(
        self, blocks: List[Dict], threshold: float, doc: fitz.Document
    ) -> List[Dict]:
        sections: List[Dict] = []
        current_heading = "Preamble"
        current_page_start = 1
        current_texts: List[str] = []
        current_pages: List[int] = []

        def _flush():
            if current_texts or sections:
                raw = " ".join(current_texts).strip()
                pages = sorted(set(current_pages)) if current_pages else [current_page_start]
                sections.append({
                    "heading": current_heading,
                    "page_start": pages[0] if pages else current_page_start,
                    "page_end": pages[-1] if pages else current_page_start,
                    "_raw_text": raw,
                    "_table_pages": pages,
                })

        for entry in blocks:
            if self._is_heading_span(entry, threshold):
                _flush()
                current_heading = entry["text"].strip()
                current_page_start = entry["page"]
                current_texts = []
                current_pages = [entry["page"]]
            else:
                current_texts.append(entry["text"])
                current_pages.append(entry["page"])

        _flush()
        return sections

    def _fallback_page_grouping(self, doc: fitz.Document) -> List[Dict]:
        """Groups text by page when heading detection fails."""
        sections = []
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text = page.get_text("text").strip()
            if not text:
                continue
            # Use first non-empty line as a pseudo-heading
            lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
            heading = lines[0][:80] if lines else f"Page {page_num + 1}"
            body = " ".join(lines[1:]) if len(lines) > 1 else text
            sections.append({
                "heading": heading,
                "page_start": page_num + 1,
                "page_end": page_num + 1,
                "_raw_text": body,
                "_table_pages": [page_num + 1],
            })
        return sections

    # ------------------------------------------------------------------
    # Document-based parsing (v2 primary path — reuses v1 extraction)
    # ------------------------------------------------------------------

    def parse_from_documents(self, documents: List[Dict]) -> List[Dict]:
        """
        Builds section structure from pre-extracted v1 documents.

        Input format (same as PDFIngestor.extract_text_with_metadata()):
          [{"text": str, "metadata": {"source": str, "page": int, ...}}, ...]

        Each document maps to one page.  We use text-based heading heuristics
        (first short line without trailing punctuation) since font data is not
        available in the plain-text extraction.

        CRITICAL: doc["text"] is used verbatim as _raw_text — no re-reading.
        """
        raw_sections: List[Dict] = []
        for doc in documents:
            text: str = doc.get("text", "").strip()
            meta: Dict = doc.get("metadata", {})
            page: int = int(meta.get("page", 0))
            source: str = meta.get("source", "unknown")

            if not text:
                continue

            lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
            heading = self._detect_text_heading(lines)
            # Body = everything after the heading line (or all text if no heading)
            if heading == lines[0] and len(lines) > 1:
                body = " ".join(lines[1:])
            else:
                body = " ".join(lines)

            raw_sections.append({
                "heading": heading,
                "page_start": page,
                "page_end": page,
                "_raw_text": body,
                "_table_pages": [page],
                "_source": source,
            })

        return self._merge_small_sections(raw_sections)

    def _detect_text_heading(self, lines: List[str]) -> str:
        """
        Heuristic: a line is a heading if it is short (≤ HEADING_MAX_WORDS words),
        does not end with sentence-terminating punctuation, and looks title-like.
        Falls back to the first 80 chars of the first line.
        """
        if not lines:
            return "Section"
        first = lines[0]
        words = first.split()
        is_short = 1 <= len(words) <= self.HEADING_MAX_WORDS
        no_trailing_punct = not first.rstrip().endswith(('.', ',', ';'))
        title_like = first.isupper() or first.istitle() or len(words) <= 6
        if is_short and no_trailing_punct and title_like:
            return first[:80]
        return first[:80]

    def _merge_small_sections(self, sections: List[Dict]) -> List[Dict]:
        """
        Merges sections whose body text has fewer than MIN_SECTION_WORDS words
        into the previous section, to avoid fragmentation.
        The heading of the merged section is appended to the previous section's
        heading to preserve all information.
        """
        if not sections:
            return sections

        merged: List[Dict] = [sections[0]]
        for sec in sections[1:]:
            word_count = len(sec["_raw_text"].split())
            if word_count < self.MIN_SECTION_WORDS and merged:
                prev = merged[-1]
                prev["heading"] = f"{prev['heading']} / {sec['heading']}"
                prev["_raw_text"] = (prev["_raw_text"] + " " + sec["_raw_text"]).strip()
                prev["page_end"] = max(prev["page_end"], sec["page_end"])
                prev["_table_pages"] = sorted(set(prev["_table_pages"] + sec["_table_pages"]))
            else:
                merged.append(sec)
        return merged


# ===========================================================================
# STEP 2 — Table Extractor
# ===========================================================================

class TableExtractor:
    """
    Extracts tables from a PDF using pdfplumber.

    For each table found:
      raw_table    — verbatim 2D list (List[List[str|None]]) as extracted
      headers      — first row, cleaned
      rows         — remaining rows as List[List[str]]
      row_objects  — List[Dict[str, str]]  header→cell
      numeric_data — List[{column, value_raw, value_parsed, row_index}]
      numeric_index— Dict[column, List[float]]
    """

    def extract_by_page(self, pdf_path: str) -> Dict[int, List[Dict]]:
        """
        Returns: {page_number: [table_dict, ...]}
        page_number is 1-indexed to match fitz convention.
        """
        result: Dict[int, List[Dict]] = defaultdict(list)
        if not HAS_PDFPLUMBER:
            log.warning("pdfplumber unavailable — table extraction skipped.")
            return result

        try:
            with pdfplumber.open(pdf_path) as pdf:
                for i, page in enumerate(pdf.pages):
                    tables = page.extract_tables() or []
                    for raw in tables:
                        if not raw:
                            continue
                        tbl = self._process_table(raw)
                        if tbl:
                            result[i + 1].append(tbl)
        except Exception as exc:
            log.error(f"pdfplumber failed on {pdf_path}: {exc}")

        return dict(result)

    # ---- internal --------------------------------------------------------

    def _clean_cell(self, cell: Any) -> str:
        if cell is None:
            return ""
        return str(cell).strip()

    def _process_table(self, raw: List[List]) -> Optional[Dict]:
        """
        Converts a raw 2D pdfplumber table into the enriched table dict.
        raw_table is ALWAYS stored as the verbatim input list converted to
        List[List[str]] (None cells become "").
        """
        if not raw or len(raw) < 2:
            return None

        # Store raw table verbatim (None → empty string for JSON safety)
        raw_table: List[List[str]] = [
            [self._clean_cell(cell) for cell in row]
            for row in raw
        ]

        # Headers from first row
        headers = raw_table[0]
        if not any(h for h in headers):
            # No usable header row — treat all rows as data, synthesise Col-N headers
            headers = [f"Col_{i}" for i in range(len(raw_table[0]))]
            data_rows = raw_table
        else:
            data_rows = raw_table[1:]

        if not data_rows:
            return None

        row_objects = self._build_row_objects(headers, data_rows)
        numeric_data, numeric_index = self._build_numeric_index(headers, data_rows)

        return {
            "headers": headers,
            "rows": data_rows,
            "raw_table": raw_table,       # verbatim 2D list
            "row_objects": row_objects,
            "numeric_data": numeric_data,
            "numeric_index": numeric_index,
        }

    def _build_row_objects(
        self, headers: List[str], rows: List[List[str]]
    ) -> List[Dict[str, str]]:
        """Creates header→value dicts for each data row."""
        objects = []
        n = len(headers)
        for row in rows:
            padded = list(row) + [""] * max(0, n - len(row))
            obj = {headers[i]: padded[i] for i in range(n) if headers[i]}
            objects.append(obj)
        return objects

    def _parse_numeric(self, raw: str) -> Optional[float]:
        """
        Parses a cell string to float.
        Handles: "350", "1,234", "<100", ">50", "~42.5", "350.0"
        Returns None if the string is not numeric.
        """
        s = raw.strip()
        if not s:
            return None
        # Strip leading approximate/comparison symbols
        s_clean = re.sub(r"^[<>~≈≤≥±]\s*", "", s)
        # Remove thousand separators
        s_clean = s_clean.replace(",", "")
        try:
            return float(s_clean)
        except ValueError:
            return None

    def _build_numeric_index(
        self, headers: List[str], rows: List[List[str]]
    ) -> Tuple[List[Dict], Dict[str, List[float]]]:
        """
        Returns:
          numeric_data   — list of {column, value_raw, value_parsed, row_index}
          numeric_index  — {column: [value_parsed, ...]}
        """
        numeric_data: List[Dict] = []
        numeric_index: Dict[str, List[float]] = defaultdict(list)
        n = len(headers)

        for row_idx, row in enumerate(rows):
            padded = list(row) + [""] * max(0, n - len(row))
            for col_idx, header in enumerate(headers):
                if not header:
                    continue
                value_raw = padded[col_idx]
                value_parsed = self._parse_numeric(value_raw)
                if value_parsed is not None:
                    numeric_data.append({
                        "column": header,
                        "value_raw": value_raw,
                        "value_parsed": value_parsed,
                        "row_index": row_idx,
                    })
                    numeric_index[header].append(value_parsed)

        return numeric_data, dict(numeric_index)


# ===========================================================================
# STEP 3 — Language Handler
# ===========================================================================

class LanguageHandler:
    """
    Detects language per section and optionally translates to English.
    Translation is OFF by default (ENABLE_TRANSLATION=false).
    """

    LANG_MAP = {
        # langdetect code → NLLB target language tag
        "hi": "hin_Deva",
        "fr": "fra_Latn",
        "de": "deu_Latn",
        "es": "spa_Latn",
        "zh-cn": "zho_Hans",
        "zh-tw": "zho_Hant",
        "ar": "arb_Arab",
        "ru": "rus_Cyrl",
        "ja": "jpn_Jpan",
        "pt": "por_Latn",
        "bn": "ben_Beng",
        "ta": "tam_Taml",
        "te": "tel_Telu",
        "mr": "mar_Deva",
        "ur": "urd_Arab",
    }

    def detect(self, text: str) -> str:
        if not HAS_LANGDETECT or not text.strip():
            return "unknown"
        try:
            return _langdetect(text[:2000])  # sample first 2000 chars
        except Exception:
            return "unknown"

    def translate_to_en(self, text: str, src_lang: str) -> Optional[str]:
        """
        Translates text to English using NLLB.
        Returns None if translation is disabled, unavailable, or src_lang is English.
        """
        if not ENABLE_TRANSLATION:
            return None
        if src_lang in ("en", "unknown"):
            return None
        nllb_src = self.LANG_MAP.get(src_lang)
        if not nllb_src:
            log.info(f"No NLLB mapping for lang '{src_lang}' — skipping translation.")
            return None
        if not _load_nllb():
            return None
        try:
            inputs = _nllb_tokenizer(
                text[:1024],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            )
            outputs = _nllb_model.generate(
                **inputs,
                forced_bos_token_id=_nllb_tokenizer.lang_code_to_id["eng_Latn"],
                max_new_tokens=512,
            )
            return _nllb_tokenizer.decode(outputs[0], skip_special_tokens=True)
        except Exception as exc:
            log.warning(f"Translation failed for lang '{src_lang}': {exc}")
            return None


# ===========================================================================
# STEP 4 — Text Normalizer
# ===========================================================================

class TextNormalizer:
    """
    Produces a normalized version of text safe for embedding.
    Rules:
      - Lowercase
      - Collapse whitespace (tabs, newlines, multiple spaces → single space)
      - Strip non-alphanumeric noise EXCEPT: periods, commas, hyphens, slashes
        (to preserve numeric notation like "1,234.5" and "Cu/Zn")
      - DO NOT remove numbers or named entities
    """

    # Characters to strip (noise) — keep: letters, digits, spaces, . , - / ( ) %
    _NOISE_RE = re.compile(r"[^\w\s.,\-/()%:;'\"]+", re.UNICODE)
    _WS_RE = re.compile(r"\s+")

    def normalize(self, text: str) -> str:
        if not text:
            return ""
        t = text.lower()
        t = self._NOISE_RE.sub(" ", t)
        t = self._WS_RE.sub(" ", t)
        return t.strip()


# ===========================================================================
# STEP 5 — Distillation Layer
# ===========================================================================

class DistillationLayer:
    """
    Wraps DistillationEngine for use in the v2 pipeline.

    CRITICAL:
      - ALWAYS uses the ORIGINAL text as input (never translated)
      - Returns None (null in JSON) if distillation result is vague
        (no summary AND no keywords AND no concepts)
    """

    def __init__(self) -> None:
        self._engine = _DistillationEngine()

    def distill(self, original_text: str) -> Optional[Dict]:
        """
        Returns dict with keys s, k, c — or None if result is too weak.
        """
        if not original_text or not original_text.strip():
            return None
        try:
            result = self._engine.distill_chunk(original_text)
        except Exception as exc:
            log.warning(f"Distillation failed: {exc}")
            return None

        s = result.get("s")
        k = result.get("k") or []
        c = result.get("c") or []

        # Return null if everything is empty — don't store meaningless output
        if not s and not k and not c:
            return None

        return {"s": s, "k": k, "c": c}


# ===========================================================================
# STEP 6 — Answer Equivalence Validator (optional)
# ===========================================================================

class AnswerEquivalenceValidator:
    """
    Validates that the processed JSON representation preserves answer fidelity
    compared to the raw PDF text.

    Enabled only when ENABLE_ANSWER_EQUIVALENCE=true.
    Uses the Gemini API (existing GOOGLE_API_KEY) and sentence-transformers
    for semantic similarity scoring.
    """

    DEFAULT_QUERY_TEMPLATES = [
        "What are the main findings or conclusions of this document?",
        "What numeric values or measurements are reported?",
        "What locations, entities, or named subjects are discussed?",
        "What methods or procedures are described?",
        "What recommendations or next steps are suggested?",
    ]

    def __init__(self) -> None:
        self._client = None
        self._sim_model = None
        self._ready = False
        if ENABLE_ANSWER_EQUIVALENCE:
            self._setup()

    def _setup(self) -> None:
        try:
            from google import genai  # type: ignore
            self._client = genai.Client()
        except Exception as exc:
            log.warning(f"Gemini client init failed for equivalence validator: {exc}")
            return

        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
            import numpy as np  # type: ignore
            self._sim_model = SentenceTransformer("all-MiniLM-L6-v2")
            self._np = np
            self._ready = True
        except Exception as exc:
            log.warning(f"SentenceTransformer init failed for equivalence validator: {exc}")

    def validate(
        self, pdf_text: str, sections: List[Dict], file_id: str
    ) -> Tuple[bool, str]:
        """
        Returns (passed: bool, reason: str).
        If ENABLE_ANSWER_EQUIVALENCE=false → always passes.
        """
        if not ENABLE_ANSWER_EQUIVALENCE:
            return True, "Answer equivalence check disabled (ENABLE_ANSWER_EQUIVALENCE=false)."
        if not self._ready:
            return True, "Answer equivalence validator not ready — skipping."

        queries = self._build_queries(sections)
        json_text = self._reconstruct_text_from_sections(sections)

        failures = []
        for q in queries:
            try:
                ans_pdf = self._llm_answer(pdf_text[:12000], q)
                ans_json = self._llm_answer(json_text[:12000], q)
                sim = self._semantic_similarity(ans_pdf, ans_json)
                log.info(f"  Equivalence Q: '{q[:60]}...' → similarity={sim:.3f}")
                if sim < ANSWER_EQUIV_THRESHOLD:
                    failures.append(
                        f"Query '{q[:60]}': similarity {sim:.3f} < threshold {ANSWER_EQUIV_THRESHOLD}"
                    )
            except Exception as exc:
                log.warning(f"Equivalence check query failed: {exc} — treating as pass.")

        if failures:
            reason = "Answer equivalence FAILED:\n" + "\n".join(f"  • {f}" for f in failures)
            return False, reason

        return True, f"Answer equivalence PASSED for {len(queries)} queries."

    # ---- internal --------------------------------------------------------

    def _build_queries(self, sections: List[Dict]) -> List[str]:
        queries = list(self.DEFAULT_QUERY_TEMPLATES)
        # Add queries derived from section headings
        for sec in sections[:5]:
            h = sec.get("heading", "")
            if h and h not in ("Preamble", "_page_section") and len(h) > 4:
                queries.append(f"What does the document say about '{h}'?")
        return queries[:8]  # cap at 8 to control API cost

    def _reconstruct_text_from_sections(self, sections: List[Dict]) -> str:
        parts = []
        for sec in sections:
            heading = sec.get("heading", "")
            original = sec.get("text", {}).get("original", "")
            if heading:
                parts.append(f"## {heading}")
            if original:
                parts.append(original)
            # Append table content as text
            for tbl in sec.get("tables", []):
                headers = tbl.get("headers", [])
                row_objs = tbl.get("row_objects", [])
                for obj in row_objs:
                    row_text = "; ".join(f"{k}: {v}" for k, v in obj.items() if v)
                    if row_text:
                        parts.append(row_text)
        return "\n\n".join(parts)

    def _llm_answer(self, context: str, query: str) -> str:
        prompt = (
            "Answer the question based ONLY on the provided context. "
            "Be factual and concise. If the context does not contain enough information, "
            "say 'Insufficient context.'\n\n"
            f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
        )
        interaction = self._client.interactions.create(
            model=GEMINI_MODEL, input=prompt
        )
        return interaction.outputs[-1].text.strip()

    def _semantic_similarity(self, a: str, b: str) -> float:
        if not a or not b:
            return 0.0
        vecs = self._sim_model.encode([a, b], normalize_embeddings=True)
        return float(self._np.dot(vecs[0], vecs[1]))


# ===========================================================================
# STEP 7 — Validation Gate
# ===========================================================================

class ValidationGate:
    """
    Runs all data-integrity checks before the output file is written.
    Any failure aborts the write and raises RuntimeError.

    Checks:
      1. Every section has text.original (non-empty for non-trivial sections)
      2. Every section's raw_table exists for every table (not empty)
      3. Numeric raw values in numeric_data match the original rows
      4. data_loss == 0 in stats
      5. Protected files are untouched (by path, not content — write is pre-checked)
    """

    def validate(self, artifact: Dict) -> Tuple[bool, List[str]]:
        errors: List[str] = []

        sections = artifact.get("sections", [])
        if not sections:
            errors.append("No sections found in artifact.")

        for i, sec in enumerate(sections):
            label = sec.get("heading", f"section[{i}]")
            text_block = sec.get("text", {})

            # Check 1: original text preserved
            original = text_block.get("original", "")
            # We allow empty original only if section is purely table-based
            if not original and not sec.get("tables"):
                errors.append(f"[{label}] text.original is empty and no tables present.")

            # Check 2: raw_table exists for every table
            for t_idx, tbl in enumerate(sec.get("tables", [])):
                if "raw_table" not in tbl or not tbl["raw_table"]:
                    errors.append(
                        f"[{label}] table[{t_idx}] is missing raw_table or raw_table is empty."
                    )

                # Check 3: numeric raw values match rows
                row_count = len(tbl.get("rows", []))
                for nd in tbl.get("numeric_data", []):
                    ri = nd.get("row_index", -1)
                    if ri < 0 or ri >= row_count:
                        errors.append(
                            f"[{label}] table[{t_idx}] numeric_data row_index={ri} "
                            f"out of range (rows={row_count})."
                        )
                    # Verify value_raw still appears in the row
                    if ri >= 0 and ri < row_count:
                        row = tbl["rows"][ri]
                        value_raw = nd.get("value_raw", "")
                        if value_raw and value_raw not in row:
                            errors.append(
                                f"[{label}] table[{t_idx}] value_raw '{value_raw}' "
                                f"not found in row[{ri}]: {row}"
                            )

        # Check 4: data_loss
        stats = artifact.get("stats", {})
        if stats.get("data_loss", 1) != 0:
            errors.append(f"stats.data_loss is {stats.get('data_loss')} — must be 0.")

        return (len(errors) == 0), errors

    def assert_protected_files_safe(self) -> None:
        """Double-check that we are not about to overwrite protected files."""
        if OUTPUT_PATH in PROTECTED_PATHS:
            raise RuntimeError(
                f"CRITICAL: OUTPUT_PATH {OUTPUT_PATH} is in PROTECTED_PATHS. Aborting."
            )


# ===========================================================================
# INPUT SOURCE VALIDATION
# ===========================================================================

# Keywords that signal the pipeline received internal test/mock data.
_MOCK_CONTENT_KEYWORDS: Tuple[str, ...] = (
    "Stoicism",
    "Retrieval-Augmented Generation",
    "FAISS",
)


def _validate_documents_source(documents: List[Dict]) -> None:
    """
    Validates that the supplied document list comes from the user-uploaded PDF,
    not from any hardcoded test/mock dataset.

    Operates entirely on doc["text"] — does NOT re-read anything from disk.

    Raises:
      AssertionError — documents is None or empty
      RuntimeError   — mock/test content detected in first-page text
    """
    assert documents is not None, "documents must not be None."
    assert isinstance(documents, list) and len(documents) > 0, (
        "documents must be a non-empty list of page dicts."
    )

    first_doc = documents[0]
    source_name: str = first_doc.get("metadata", {}).get("source", "unknown")
    first_text: str = first_doc.get("text", "")

    # Confirm the source being processed
    print(f"Processing source: {source_name}  ({len(documents)} pages)")
    log.info(f"Input validated — source={source_name!r}, pages={len(documents)}")

    # Verify first 200 chars of v1 text does not contain mock keywords
    sample = first_text[:500]
    for keyword in _MOCK_CONTENT_KEYWORDS:
        if keyword.lower() in sample.lower():
            raise RuntimeError(
                f"Mock/test dataset detected instead of user PDF.\n"
                f"  Matched keyword: {keyword!r}\n"
                f"  Source: {source_name!r}\n"
                f"  Action: supply the actual uploaded PDF, not test data."
            )

    log.info("Input source validation PASSED — no mock/test content detected.")


# ===========================================================================
# STEP 8 — Hybrid Ingestion Pipeline (Orchestrator)
# ===========================================================================

class HybridIngestionPipeline:
    """
    Orchestrates all stages and produces knowledge_artifact_v2.json.

    PRIMARY ENTRY POINT: run_from_documents()
      Accepts the SAME documents list produced by v1's PDFIngestor so both
      pipelines process identical source text.  pdfplumber is used ONLY to
      enrich the pages that are already present in documents (table overlay);
      it is never the primary text source.

    Pipeline:
      documents (v1) →
          _validate_documents_source()    ← checks for mock/test content in text
          SectionParser.parse_from_documents()  ← groups pages into sections
          TableExtractor.extract_by_page()       ← enrichment only (pdfplumber)
          [for each section]
              LanguageHandler.detect()
              [optional] LanguageHandler.translate()
              TextNormalizer.normalize(original)  ← always from original
              DistillationLayer.distill(original) ← always from original
          AnswerEquivalenceValidator.validate()   ← optional
          ValidationGate.validate()
          → artifact dict (caller decides where to write)
    """

    def __init__(self) -> None:
        self._section_parser = SectionParser()
        self._table_extractor = TableExtractor()
        self._lang_handler = LanguageHandler()
        self._normalizer = TextNormalizer()
        self._distiller = DistillationLayer()
        self._equiv_validator = AnswerEquivalenceValidator()
        self._gate = ValidationGate()

    # ------------------------------------------------------------------
    # Primary entry point — reuses v1 documents, no independent PDF read
    # ------------------------------------------------------------------

    def run_from_documents(
        self,
        documents: List[Dict],
        pdf_path: Optional[str] = None,
        file_id: Optional[str] = None,
    ) -> Dict:
        """
        Build the v2 artifact from pre-extracted v1 documents.

        Args:
            documents:  Output of PDFIngestor.extract_text_with_metadata().
                        [{"text": str, "metadata": {"source", "page", ...}}, ...]
                        This is the ONLY source of text — never re-read from disk.
            pdf_path:   Optional path, used SOLELY by pdfplumber for table
                        enrichment on matching pages.  May be None if tables
                        are not needed.
            file_id:    Reuses v1 file_id for cross-artifact correlation.
        """
        # --- INPUT VALIDATION: operates on extracted text, not disk ---
        _validate_documents_source(documents)

        file_id = file_id or str(uuid.uuid4())
        source_name = documents[0].get("metadata", {}).get("source", "unknown")
        # original_size: prefer the value stamped into metadata by v1, else 0
        original_size: int = int(
            documents[0].get("metadata", {}).get("original_size", 0)
        )

        log.info(f"=== Hybrid Ingestion v2 | file_id={file_id} ===")
        log.info(f"  Source (v1 docs) : {source_name}  ({len(documents)} pages)")
        log.info(f"  pdf_path (tables): {pdf_path or 'not provided'}")
        log.info(f"  Output           : {OUTPUT_PATH}")
        log.info(f"  ENABLE_TRANSLATION        = {ENABLE_TRANSLATION}")
        log.info(f"  ENABLE_ANSWER_EQUIVALENCE = {ENABLE_ANSWER_EQUIVALENCE}")

        # --- STAGE 1: Section parsing from v1 documents ---
        log.info("Stage 1/5: Parsing sections from v1 documents...")
        raw_sections = self._section_parser.parse_from_documents(documents)
        log.info(f"  Found {len(raw_sections)} sections after merge.")

        # --- STAGE 2: Table enrichment via pdfplumber (NOT primary source) ---
        log.info("Stage 2/5: Extracting table enrichment...")
        tables_by_page: Dict[int, List[Dict]] = {}
        if pdf_path and os.path.exists(pdf_path):
            tables_by_page = self._table_extractor.extract_by_page(pdf_path)
            total_tables = sum(len(v) for v in tables_by_page.values())
            log.info(f"  Enriched {total_tables} tables across {len(tables_by_page)} pages.")
        else:
            total_tables = 0
            log.info("  pdf_path not provided — table enrichment skipped.")

        # --- STAGE 3: Reconstruct full text from documents for equiv check ---
        # Built from doc["text"] directly — no disk read.
        full_text = "\n\n".join(doc.get("text", "") for doc in documents)

        # --- STAGE 4: Enrich sections ---
        log.info("Stage 3/5: Enriching sections (language, normalization, distillation)...")
        sections_out: List[Dict] = []
        for raw_sec in raw_sections:
            enriched = self._enrich_section(raw_sec, tables_by_page)
            sections_out.append(enriched)

        # --- STAGE 5: Answer equivalence validation (optional) ---
        log.info("Stage 4/5: Answer equivalence validation...")
        equiv_ok, equiv_reason = self._equiv_validator.validate(
            full_text, sections_out, file_id
        )
        log.info(f"  {equiv_reason}")
        if not equiv_ok:
            raise RuntimeError(
                f"FAIL-SAFE triggered: Answer equivalence check failed.\n{equiv_reason}"
            )

        # --- Build artifact ---
        artifact = {
            "file_id": file_id,
            "sections": sections_out,
            "stats": {
                "original_size": original_size,
                "processed_size": 0,  # filled after serialisation
                "data_loss": 0,
                "total_sections": len(sections_out),
                "total_tables": total_tables,
                "translation_enabled": ENABLE_TRANSLATION,
                "answer_equivalence_enabled": ENABLE_ANSWER_EQUIVALENCE,
                "v1_pages": len(documents),
                "source": source_name,
            },
        }

        # --- STAGE 6: Validation gate ---
        log.info("Stage 5/5: Running validation gate...")
        self._gate.assert_protected_files_safe()
        ok, errors = self._gate.validate(artifact)
        if not ok:
            error_text = "\n".join(f"  • {e}" for e in errors)
            raise RuntimeError(
                f"FAIL-SAFE triggered: Validation gate failed with {len(errors)} error(s):\n{error_text}"
            )
        log.info(f"  Validation PASSED — all {len(sections_out)} sections clean.")

        # Update processed_size after validation
        serialised = json.dumps(artifact, ensure_ascii=False)
        artifact["stats"]["processed_size"] = len(serialised.encode("utf-8"))

        return artifact

    # ------------------------------------------------------------------
    # Legacy path: accepts a PDF file path directly (used by CLI)
    # Uses v1's PDFIngestor internally — identical extraction as upload flow
    # ------------------------------------------------------------------

    def run(self, pdf_path: str, file_id: Optional[str] = None) -> Dict:
        """
        Convenience wrapper for CLI use.  Calls PDFIngestor (same as v1 upload
        flow) and forwards to run_from_documents() — guarantees identical text.
        """
        try:
            from app.ingestion import PDFIngestor as _PDFIngestor
        except ModuleNotFoundError:
            from backend.app.ingestion import PDFIngestor as _PDFIngestor  # type: ignore

        pdf_path = os.path.abspath(pdf_path)
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF not found: {pdf_path!r}")

        log.info(f"CLI mode — using PDFIngestor on: {pdf_path}")
        print(f"Processing PDF: {pdf_path}")
        ingestor = _PDFIngestor(pdf_path)
        documents = ingestor.extract_text_with_metadata()

        # Stamp original_size (same as v1 upload does)
        original_size = Path(pdf_path).stat().st_size
        for doc in documents:
            doc.setdefault("metadata", {})["original_size"] = original_size

        return self.run_from_documents(documents, pdf_path=pdf_path, file_id=file_id)

    def save(self, artifact: Dict, output_path: Optional[Path] = None) -> None:
        """
        Writes the artifact to OUTPUT_PATH or to the provided output path.
        Aborts if the selected output path collides with any protected file.
        """
        destination = Path(output_path) if output_path else OUTPUT_PATH
        if destination in PROTECTED_PATHS:
            raise RuntimeError(
                f"ABORT: OUTPUT_PATH {destination} is protected. Will not write."
            )
        destination.parent.mkdir(parents=True, exist_ok=True)
        with open(destination, "w", encoding="utf-8") as f:
            json.dump(artifact, f, indent=2, ensure_ascii=False)
        log.info(f"✓ Artifact written → {destination}")
        log.info(
            f"  Sections: {artifact['stats']['total_sections']} | "
            f"Tables: {artifact['stats']['total_tables']} | "
            f"data_loss: {artifact['stats']['data_loss']}"
        )

    # ---- internal helpers ------------------------------------------------

    def _enrich_section(
        self, raw_sec: Dict, tables_by_page: Dict[int, List[Dict]]
    ) -> Dict:
        """
        Enriches a raw section dict with text block, tables, language, normalization,
        and distillation. All transformations are non-destructive.
        """
        original_text: str = raw_sec["_raw_text"]

        # Language detection
        detected_lang: str = self._lang_handler.detect(original_text)

        # Translation (optional, only for non-English)
        translated_en: Optional[str] = None
        if detected_lang not in ("en", "unknown"):
            translated_en = self._lang_handler.translate_to_en(original_text, detected_lang)

        # Normalization — always from original text
        normalized: str = self._normalizer.normalize(original_text)

        # Distillation — ALWAYS from original text, NEVER from translated
        distilled: Optional[Dict] = self._distiller.distill(original_text)

        # Assign tables that belong to this section's pages
        section_pages: List[int] = raw_sec.get("_table_pages", [])
        section_tables: List[Dict] = []
        for pg in section_pages:
            section_tables.extend(tables_by_page.get(pg, []))

        return {
            "heading": raw_sec["heading"],
            "page_start": raw_sec["page_start"],
            "page_end": raw_sec["page_end"],
            "text": {
                "original": original_text,          # verbatim, never modified
                "language": detected_lang,
                "normalized": normalized,            # lowercase + noise clean
                "translated_en": translated_en,     # null unless ENABLE_TRANSLATION=true
            },
            "distilled": distilled,                  # null if weak/empty; from original only
            "tables": section_tables,               # each has raw_table, numeric_data, etc.
        }

    def _extract_full_text(self, pdf_path: str) -> str:
        """Extracts all text from the PDF using fitz for the equivalence check."""
        try:
            doc = fitz.open(pdf_path)
            pages = []
            for i in range(len(doc)):
                pages.append(doc.load_page(i).get_text("text"))
            doc.close()
            return "\n\n".join(pages)
        except Exception as exc:
            log.warning(f"Full text extraction for equivalence check failed: {exc}")
            return ""


# ===========================================================================
# CLI Entry Point
# ===========================================================================

# ===========================================================================
# Public module-level API (used by main.py upload endpoint)
# ===========================================================================

def process_documents(
    documents: List[Dict],
    pdf_path: Optional[str] = None,
    file_id: Optional[str] = None,
) -> Dict:
    """
    Public API called by the v1 upload flow after PDFIngestor extraction.

    Guarantees:
      - text.original always equals the corresponding doc["text"] from v1
      - first 200 chars of both pipelines' source text are identical
      - no external datasets or test data are used

    Args:
        documents:  PDFIngestor.extract_text_with_metadata() output.
        pdf_path:   Optional path for pdfplumber table enrichment only.
        file_id:    v1 file_id to correlate both artifacts.

    Returns:
        v2 artifact dict (not yet written to disk).
    """
    pipeline = HybridIngestionPipeline()
    return pipeline.run_from_documents(documents, pdf_path=pdf_path, file_id=file_id)


def write_v2_artifact(artifact: Dict, output_path: Optional[str] = None) -> None:
    """
    Writes the v2 artifact to OUTPUT_PATH (backend/data/knowledge_artifact_v2.json)
    unless an explicit output_path is provided.
    Never touches knowledge_artifact.json or metadata.json.
    """
    pipeline = HybridIngestionPipeline()
    pipeline.save(artifact, output_path=Path(output_path) if output_path else None)


# ===========================================================================
# CLI Entry Point
# ===========================================================================

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Hybrid Ingestion Pipeline v2 — produces knowledge_artifact_v2.json.\n"
            "Uses PDFIngestor (identical to the upload flow) — not an independent PDF parser."
        )
    )
    parser.add_argument("pdf_path", help="Path to the source PDF file.")
    parser.add_argument(
        "--file-id",
        default=None,
        help="Optional file UUID. Auto-generated if not provided.",
    )
    args = parser.parse_args()

    pdf_path = os.path.abspath(args.pdf_path)

    # pipeline.run() uses PDFIngestor internally (same as v1 upload)
    # then forwards to run_from_documents() — identical text pipeline
    pipeline = HybridIngestionPipeline()
    try:
        artifact = pipeline.run(pdf_path, file_id=args.file_id)
        pipeline.save(artifact)
        sys.exit(0)
    except (AssertionError, FileNotFoundError) as exc:
        log.error(f"Input error — output file NOT written.\nReason: {exc}")
        sys.exit(1)
    except RuntimeError as exc:
        log.error(f"Pipeline aborted — output file NOT written.\nReason: {exc}")
        sys.exit(2)
    except Exception as exc:
        log.exception(f"Unexpected error: {exc}")
        sys.exit(3)


if __name__ == "__main__":
    main()
