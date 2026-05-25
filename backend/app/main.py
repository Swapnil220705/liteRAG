from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import json
import shutil
import os
import uuid
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
ARTIFACT_DIR = DATA_DIR / "artifacts"

load_dotenv(BASE_DIR / ".env")

from app.ingestion import PDFIngestor
from app.chunking import SemanticChunker
from app.distillation import DistillationEngine
from app.embeddings import EmbeddingEngine
from app.retrieval import VectorStore
from app.optimization import ContextOptimizer
from app.cache import QueryCache
from app.generation import AnswerGenerator
from app.query_processing import QueryProcessor
from app.logging_utils import log_event
from app.hybrid_ingestion import process_documents as hybrid_process_documents, write_v2_artifact
from app.artifact_store import ArtifactStore

app = FastAPI(title="liteRAG API")

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Content-Disposition"],
)

# Initialize components
embedding_engine = EmbeddingEngine()
try:
    vector_store = VectorStore(dimension=embedding_engine.dimension)
except RuntimeError as exc:
    print(f"Vector store load failed during startup: {exc}")
    vector_store = VectorStore(dimension=embedding_engine.dimension, auto_load=False)
optimizer = ContextOptimizer(token_limit=800)
cache = QueryCache()
generator = AnswerGenerator()
query_processor = QueryProcessor()

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
artifact_store = ArtifactStore()

class QueryRequest(BaseModel):
    query: str


def _tokenize_for_confidence(text: str) -> set[str]:
    return {token for token in "".join(char.lower() if char.isalnum() else " " for char in text).split() if token}


def build_confidence_payload(query: str, query_type: str, retrieved_chunks):
    if not retrieved_chunks:
        return {
            "level": "low",
            "score": 0.0,
            "max_score": 0.0,
            "avg_score": 0.0,
            "max_cross_encoder_score": 0.0,
            "score_gap": 0.0,
            "consistency": 0.0,
            "lexical_support": 0.0,
            "sources": 0,
        }

    rerank_scores = [chunk.get("rerank_score", chunk.get("score", 0.0)) for chunk in retrieved_chunks]
    dense_scores = [chunk.get("dense_score", 0.0) for chunk in retrieved_chunks]
    keyword_scores = [chunk.get("keyword_score", 0.0) for chunk in retrieved_chunks]
    cross_scores = [chunk.get("cross_encoder_score", 0.0) for chunk in retrieved_chunks]
    sorted_scores = sorted(rerank_scores, reverse=True)
    max_score = max(rerank_scores)
    avg_score = sum(rerank_scores) / len(rerank_scores)
    max_cross_score = max(cross_scores)
    score_gap = sorted_scores[0] - sorted_scores[1] if len(sorted_scores) > 1 else sorted_scores[0]
    score_spread = max_score - min(rerank_scores) if len(rerank_scores) > 1 else 0.0
    consistency = max(0.0, 1.0 - score_spread)
    query_tokens = _tokenize_for_confidence(query)
    lexical_support_scores = []
    for chunk in retrieved_chunks:
        candidate_tokens = _tokenize_for_confidence(chunk.get("text", ""))
        overlap = len(query_tokens.intersection(candidate_tokens))
        lexical_support_scores.append(overlap / max(len(query_tokens), 1) if query_tokens else 0.0)

    lexical_support = sum(lexical_support_scores) / max(len(lexical_support_scores), 1)
    dense_support = sum(dense_scores) / len(dense_scores)
    keyword_support = sum(keyword_scores) / len(keyword_scores)
    source_factor = min(len(retrieved_chunks) / 3.0, 1.0)

    if query_type == "summarization":
        combined_score = (
            avg_score * 0.30
            + dense_support * 0.15
            + keyword_support * 0.20
            + lexical_support * 0.15
            + consistency * 0.10
            + source_factor * 0.10
        )
    else:
        combined_score = (
            max_score * 0.30
            + avg_score * 0.15
            + max_cross_score * 0.20
            + min(score_gap, 1.0) * 0.10
            + lexical_support * 0.10
            + consistency * 0.10
            + source_factor * 0.05
        )

    if max_cross_score < 0.05 and lexical_support < 0.12 and keyword_support < 0.18:
        level = "low"
    elif query_type == "summarization":
        if combined_score >= 0.42 and lexical_support >= 0.18:
            level = "high"
        elif combined_score >= 0.26 and lexical_support >= 0.1:
            level = "medium"
        else:
            level = "low"
    elif combined_score >= 0.52 and max_cross_score >= 0.35 and score_gap >= 0.08:
        level = "high"
    elif combined_score >= 0.34 and (max_cross_score >= 0.12 or lexical_support >= 0.2):
        level = "medium"
    else:
        level = "low"


    return {
        "level": level,
        "score": round(combined_score, 4),
        "max_score": round(max_score, 4),
        "avg_score": round(avg_score, 4),
        "max_cross_encoder_score": round(max_cross_score, 4),
        "score_gap": round(score_gap, 4),
        "consistency": round(consistency, 4),
        "lexical_support": round(lexical_support, 4),
        "sources": len(retrieved_chunks),
    }


def apply_confidence_overrides(confidence: dict, retrieval_telemetry: dict) -> dict:
    adjusted = dict(confidence)
    if retrieval_telemetry.get("soft_fallback_used"):
        adjusted["level"] = "low"
        adjusted["score"] = min(adjusted.get("score", 0.0), 0.29)
    return adjusted


def should_skip_llm(confidence: dict, context_package: dict, query_type: str) -> bool:
    min_token_budget = 55 if query_type == "summarization" else 40
    weak_budget = 24 if query_type == "summarization" else 18
    token_estimate = context_package.get("token_estimate", 0)
    if token_estimate < weak_budget:
        return True
    if confidence["level"] == "low":
        return False
    min_sentences = 2 if query_type in {"summarization", "analytical"} else 1
    selected_count = len(context_package.get("selected_sentences", []))
    if selected_count == 0:
        return True
    if token_estimate < min_token_budget and confidence["level"] != "high":
        return True
    if selected_count < min_sentences and confidence["level"] != "high":
        return True
    if query_type == "analytical" and confidence["level"] == "low" and selected_count < 2:
        return True
    return False


def apply_uncertainty_prefix(answer: str, confidence: dict) -> str:
    if confidence.get("level") != "low":
        return answer
    uncertainty_prefix = "Low-confidence answer based on partial context: "
    if answer.startswith(uncertainty_prefix):
        return answer
    return f"{uncertainty_prefix}{answer}"


def build_sources_payload(retrieved_chunks):
    return [{
        "text": c.get("text", ""),
        "score": c.get("score", 0.0),
        "dense_score": c.get("dense_score", 0.0),
        "keyword_score": c.get("keyword_score", 0.0),
        "rerank_score": c.get("rerank_score", 0.0),
        "rank": c.get("rank", i + 1),
        **c.get("metadata", {})
    } for i, c in enumerate(retrieved_chunks)]


def build_direct_answer(query_type: str, context_package: dict, retrieved_chunks):
    selected_sentences = context_package.get("selected_sentences", [])
    if selected_sentences:
        sentence_count = 2 if query_type == "definition" else 3
        answer_text = " ".join(item["text"] for item in selected_sentences[:sentence_count]).strip()
        if answer_text:
            return f"(Direct Extract) {answer_text}"

    top_chunk_text = retrieved_chunks[0].get("text", "") if retrieved_chunks else ""
    return f"(Direct Extract) {top_chunk_text}".strip()


def format_distilled_context(chunk: dict) -> str:
    distilled = chunk.get("distilled") or {}
    if not isinstance(distilled, dict):
        return ""

    parts = []
    summary = distilled.get("s")
    if isinstance(summary, str) and summary.strip():
        parts.append(summary.strip())

    concepts = distilled.get("c") or []
    formatted_concepts = []
    for concept in concepts:
        if isinstance(concept, list) and len(concept) == 3:
            subject, relation, obj = (str(part).strip() for part in concept)
            if subject and relation and obj:
                formatted_concepts.append(f"{subject} {relation} {obj}")

    if formatted_concepts:
        parts.append(f"Concepts: {'; '.join(formatted_concepts)}.")

    return " ".join(parts).strip()


def build_weak_context_package(retrieved_chunks, query_type: str) -> dict:
    fallback_k = 1 if query_type in {"definition", "fact_lookup"} else 2
    fallback_chunks = retrieved_chunks[:fallback_k]
    pages = []
    fragments = []
    for chunk in fallback_chunks:
        page = chunk.get("metadata", {}).get("page")
        text = format_distilled_context(chunk) or chunk.get("text", "").strip()
        if page is not None:
            pages.append(page)
            fragments.append(f"[Page {page}] {text}")
        elif text:
            fragments.append(text)

    context = "\n".join(fragment for fragment in fragments if fragment).strip()
    token_estimate = len(context) // 4 if context else 0
    return {
        "context": context,
        "selected_sentences": [],
        "token_estimate": token_estimate,
        "pages_used": sorted(set(pages)),
        "relevance_floor": None,
        "reasoning_summary": (
            f"Used {len(fallback_chunks)} weakly matched fallback chunk(s) because sentence-level compression did not retain enough context."
        ),
    }


def write_knowledge_artifact(file_id: str, chunks, artifact_path: Path | str | None = None) -> Path:
    original_size = None
    if chunks:
        original_size = chunks[0].get("metadata", {}).get("original_size")

    artifact = {
        "file_id": file_id,
        "original_size": original_size,
        "total_chunks": len(chunks),
        "chunks": [
            {
                "id": i,
                "page": chunk.get("metadata", {}).get("page"),
                "distilled": chunk.get("distilled", {}),
            }
            for i, chunk in enumerate(chunks)
        ],
    }

    destination = Path(artifact_path or ARTIFACT_DIR / f"{file_id}_v1.json")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with open(destination, "w", encoding="utf-8") as f:
        json.dump(artifact, f, indent=2)
    return destination


def clear_knowledge_artifact() -> None:
    legacy_path = DATA_DIR / "knowledge_artifact.json"
    if legacy_path.exists():
        legacy_path.unlink()


def get_artifact_status_payload(file_id: str | None = None) -> dict:
    if file_id:
        artifact = artifact_store.get_artifact(file_id)
        if not artifact:
            return {
                "ready": False,
                "file_id": file_id,
            }

        return {
            "ready": True,
            "artifact": artifact,
        }

    artifacts = artifact_store.list_artifacts()
    return {
        "ready": bool(artifacts),
        "count": len(artifacts),
        "artifacts": artifacts,
    }

@app.get("/status")
async def get_status():
    is_indexed = vector_store.has_documents()
    latest_metadata = vector_store.metadata[0]["metadata"] if is_indexed and vector_store.metadata else {}

    return {
        "indexed": is_indexed,
        "chunks": len(vector_store.metadata),
        "source": latest_metadata.get("source"),
        "pages": latest_metadata.get("total_pages"),
    }

@app.post("/reset")
async def reset_session():
    try:
        vector_store.reset()
        cache.clear()
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Failed to reset session state.") from exc

    return {"message": "Session reset successfully", "indexed": False}


@app.get("/artifact/status")
async def artifact_status(file_id: str | None = None):
    return get_artifact_status_payload(file_id=file_id)


@app.get("/artifacts")
async def list_artifacts():
    return {"artifacts": artifact_store.list_artifacts()}


@app.get("/artifact/{file_id}")
async def get_artifact(file_id: str):
    artifact = artifact_store.get_artifact(file_id)
    if not artifact:
        raise HTTPException(status_code=404, detail="Artifact not found.")
    return artifact


@app.get("/artifact/download/{file_id}")
def download_artifact(file_id: str, version: str = "v2"):
    artifact = artifact_store.get_artifact(file_id)
    if not artifact:
        raise HTTPException(status_code=404, detail="Artifact not found.")

    path_str = artifact.get("artifact_v2_path") if version == "v2" else artifact.get("artifact_v1_path")
    if not path_str:
        raise HTTPException(status_code=404, detail=f"Artifact version '{version}' is not available.")

    file_path = Path(path_str)
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Artifact file not found.")

    return FileResponse(
        path=file_path,
        filename=file_path.name,
        media_type="application/json",
    )


@app.get("/export")
def export_knowledge(file_id: str, version: str = "v2"):
    artifact = artifact_store.get_artifact(file_id)
    if not artifact:
        raise HTTPException(status_code=404, detail="Artifact not found.")

    path_str = artifact.get("artifact_v2_path") if version == "v2" else artifact.get("artifact_v1_path")
    if not path_str:
        raise HTTPException(status_code=404, detail=f"Artifact version '{version}' is not available.")

    file_path = Path(path_str)
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Artifact file not found.")

    return FileResponse(
        path=file_path,
        filename=file_path.name,
        media_type="application/json",
    )

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed.")
    
    file_id = str(uuid.uuid4())
    file_path = UPLOAD_DIR / f"{file_id}.pdf"

    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Failed to save uploaded PDF.") from exc
    
    # Processing stages for progress feedback (emulated via logging for now)
    print(f"[{file_id}] Stage: Extracting text...")
    log_event("upload_started", file_id=file_id, filename=file.filename)
    try:
        ingestor = PDFIngestor(str(file_path))
        docs = ingestor.extract_text_with_metadata()
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Failed to extract text from PDF.") from exc
    
    print(f"[{file_id}] Stage: Chunking...")
    chunker = SemanticChunker()
    chunks = chunker.chunk_documents(docs)

    print(f"[{file_id}] Stage: Distilling chunks...")
    distiller = DistillationEngine()
    original_size = file_path.stat().st_size
    for chunk in chunks:
        chunk.setdefault("metadata", {})
        chunk["metadata"]["original_size"] = original_size
        try:
            chunk["distilled"] = distiller.distill_chunk(chunk.get("text", ""))
        except Exception:
            chunk["distilled"] = {"s": "", "k": [], "c": []}
    log_event("upload_distilled", file_id=file_id, chunks=len(chunks))
    
    print(f"[{file_id}] Stage: Generating embeddings...")
    try:
        texts = [c["text"] for c in chunks]
        embeddings = embedding_engine.generate_embeddings(texts)
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Failed to generate embeddings.") from exc
    
    print(f"[{file_id}] Stage: Indexing...")
    try:
        vector_store.clear()
        vector_store.add_documents(embeddings, chunks)
        vector_store.save()
        v1_path = write_knowledge_artifact(file_id, chunks)
        log_event("upload_indexed", file_id=file_id, pages=len(docs), chunks=len(chunks))
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Failed to index document.") from exc
    
    # Invalidate cache on new upload
    try:
        cache.clear()
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Failed to clear query cache.") from exc

    v2_path = None
    v2_size = None
    try:
        print(f"[{file_id}] Stage: Building v2 knowledge artifact...")
        v2_artifact = hybrid_process_documents(
            documents=docs,
            pdf_path=str(file_path),   # pdfplumber table enrichment only
            file_id=file_id,           # correlate with v1 artifact
        )
        v2_path = ARTIFACT_DIR / f"{file_id}_v2.json"
        write_v2_artifact(v2_artifact, output_path=str(v2_path))
        v2_size = v2_path.stat().st_size
        log_event("upload_v2_indexed", file_id=file_id,
                  sections=v2_artifact["stats"]["total_sections"],
                  tables=v2_artifact["stats"]["total_tables"])
    except Exception as exc:
        print(f"[{file_id}] WARNING: v2 ingestion failed (non-fatal): {exc}")
        log_event("upload_v2_failed", file_id=file_id, error=str(exc))

    try:
        artifact_store.add_artifact(
            file_id=file_id,
            filename=file.filename,
            source=docs[0].get("metadata", {}).get("source"),
            pages=len(docs),
            original_size=original_size,
            artifact_v1_path=str(v1_path),
            artifact_v1_size=v1_path.stat().st_size,
            artifact_v2_path=str(v2_path) if v2_path else None,
            artifact_v2_size=v2_size,
            status="ready",
            metadata={"v2_created": bool(v2_path)},
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Failed to save artifact metadata.") from exc

    return {
        "message": "PDF processed and indexed successfully",
        "file_id": file_id,
        "pages": len(docs),
        "artifact_v1_path": str(v1_path),
        "artifact_v1_size": v1_path.stat().st_size,
        "artifact_v2_path": str(v2_path) if v2_path else None,
        "artifact_v2_size": v2_size,
    }

@app.post("/query")
async def query_document(request: QueryRequest):
    query = request.query
    query_package = query_processor.build_query_package(query)
    retrieval_query = query_package["embedding_query"] or query_package["rewritten"] or query_package["normalized"] or query
    retrieval_text = query_package["keyword_query"] or query.strip()
    
    try:
        query_emb = embedding_engine.generate_single_embedding(retrieval_query)
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Failed to generate query embedding.") from exc

    # 1. Check Cache
    try:
        cached_response = cache.get(query, query_embedding=query_emb)
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Failed to read query cache.") from exc

    if cached_response and cached_response.get("answer"):
        print(f"Cache hit -> skipping LLM")
        log_event(
            "query_cache_hit",
            query_type=query_package["query_type"],
            cache_match=cached_response.get("cache_match", "exact"),
            cache_similarity=cached_response.get("cache_similarity"),
            llm_called=False,
        )
        return {
            "answer": cached_response.get("answer"),
            "sources": cached_response.get("sources", []),
            "confidence": cached_response.get("confidence", {"level": "medium", "score": 0.5}),
            "reasoning_summary": cached_response.get("reasoning_summary", "Served from cache."),
            "pages_referenced": cached_response.get("pages_referenced", []),
            "query_type": cached_response.get("query_type", query_package["query_type"]),
            "cached": True,
        }
    
    # 2. Retrieve
    try:
        retrieved_chunks, retrieval_telemetry = vector_store.hybrid_search(
            query_text=retrieval_text,
            query_embedding=query_emb,
            initial_k=10,
            final_k=3,
            threshold=0.12,
            query_type=query_package["query_type"],
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Failed to search vector store.") from exc

    print(f"Chunks retrieved: {len(retrieved_chunks)}")
    confidence = build_confidence_payload(query, query_package["query_type"], retrieved_chunks)
    confidence = apply_confidence_overrides(confidence, retrieval_telemetry)
    sources_payload = build_sources_payload(retrieved_chunks)
    log_event(
        "query_retrieval_complete",
        query_type=query_package["query_type"],
        classification_confidence=query_package["classification_confidence"],
        classification_signals=query_package["classification_signals"],
        normalized_query=query_package["normalized"],
        rewritten_query=query_package["rewritten"],
        confidence=confidence,
        telemetry=retrieval_telemetry,
    )
    log_event(
        "retrieval_evaluation",
        query_type=query_package["query_type"],
        metrics=retrieval_telemetry.get("retrieval_metrics", {}),
        selected_chunk_ids=retrieval_telemetry.get("selected_chunk_ids", []),
        discarded_candidates=retrieval_telemetry.get("discarded_candidates", []),
    )
    
    if not retrieved_chunks:
        print("No relevant context -> skipping LLM")
        return {
            "answer": "I don't know the answer to this as the document does not contain relevant context.",
            "sources": [],
            "confidence": confidence,
            "reasoning_summary": "No retrieval candidates cleared the hybrid retrieval threshold.",
            "pages_referenced": [],
            "query_type": query_package["query_type"],
            "cached": False
        }

    # 4. Optimize Context
    context_package = optimizer.build_context_package(query, retrieved_chunks, query_type=query_package["query_type"])
    context = context_package["context"]
    
    if not context or len(context.strip()) < 10:
        if retrieved_chunks:
            context_package = build_weak_context_package(retrieved_chunks, query_package["query_type"])
            context = context_package["context"]
            confidence["level"] = "low"
            confidence["score"] = min(confidence.get("score", 0.0), 0.25)
        else:
            print("No relevant context -> skipping LLM")
            return {
                "answer": "I don't know the answer to this as the document does not contain relevant context.",
                "sources": [],
                "confidence": confidence,
                "reasoning_summary": "Context compression did not produce enough grounded content for answer generation.",
                "pages_referenced": [],
                "query_type": query_package["query_type"],
                "cached": False
            }

    final_context_size = context_package["token_estimate"]
    print(f"Final context size: ~{final_context_size} tokens")
    log_event(
        "context_optimized",
        query_type=query_package["query_type"],
        token_estimate=final_context_size,
        selected_sentences=len(context_package.get("selected_sentences", [])),
        pages_used=context_package.get("pages_used", []),
        relevance_floor=context_package.get("relevance_floor"),
    )

    reasoning_summary = context_package["reasoning_summary"]
    pages_referenced = context_package["pages_used"]
    llm_called = False

    if retrieval_telemetry.get("soft_fallback_used"):
        reasoning_summary = (
            f"{reasoning_summary} "
            f"Used a soft fallback over the top reranked candidates because no chunk cleared the final threshold."
        )

    # 5. Adaptive generation path
    if (
        query_package["should_prefer_direct_answer"]
        and not query_package["force_full_rag"]
        and confidence["level"] == "high"
        and confidence["max_cross_encoder_score"] >= 0.45
    ):
        print(f"Simple query '{query}' -> returning direct grounded extract, skipping LLM")
        answer = build_direct_answer(query_package["query_type"], context_package, retrieved_chunks)
        reasoning_summary = (
            f"{reasoning_summary} "
            f"Skipped LLM because the query was classified as {query_package['query_type']} "
            f"and the retrieval confidence was {confidence['level']}."
        )
    else:
        if should_skip_llm(confidence, context_package, query_package["query_type"]):
            log_event(
                "query_skipped_llm",
                query_type=query_package["query_type"],
                reason="weak_context",
                confidence=confidence,
                token_estimate=final_context_size,
            )
            return {
                "answer": "I don't know the answer to this with enough confidence from the indexed document. Please try a more specific query.",
                "sources": sources_payload,
                "confidence": confidence,
                "reasoning_summary": "LLM was skipped because the retrieved context was too weak for a grounded answer.",
                "pages_referenced": pages_referenced,
                "query_type": query_package["query_type"],
                "cached": False,
            }

        print("Calling LLM")
        try:
            answer = generator.generate_answer(query, context)
            llm_called = True
            reasoning_summary = (
                f"{reasoning_summary} "
                f"Used Gemini after hybrid retrieval, reranking, and sentence-level compression."
            )
            if confidence["level"] == "low":
                reasoning_summary = (
                    f"{reasoning_summary} "
                    f"The answer is low-confidence because it was generated from partial or weakly matched context."
                )
        except Exception as exc:
            raise HTTPException(status_code=500, detail="Failed to generate answer from LLM.") from exc

    answer = apply_uncertainty_prefix(answer, confidence)

    response_payload = {
        "answer": answer,
        "sources": sources_payload,
        "confidence": confidence,
        "reasoning_summary": reasoning_summary,
        "pages_referenced": pages_referenced,
        "query_type": query_package["query_type"],
    }

    # 6. Cache & Return
    try:
        cache.set(query, response_payload, query_embedding=query_emb)
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Failed to update query cache.") from exc

    log_event(
        "query_completed",
        query_type=query_package["query_type"],
        confidence=confidence,
        token_estimate=final_context_size,
        llm_called=llm_called,
        sources=len(sources_payload),
    )
    
    return {
        "answer": answer,
        "sources": sources_payload,
        "confidence": confidence,
        "reasoning_summary": reasoning_summary,
        "pages_referenced": pages_referenced,
        "query_type": query_package["query_type"],
        "cached": False
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
