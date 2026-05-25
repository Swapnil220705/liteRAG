import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DEFAULT_DB_PATH = DATA_DIR / "artifact_store.db"


class ArtifactStore:
    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = Path(db_path or DEFAULT_DB_PATH)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = self._connect()
        self._create_tables()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(str(self.db_path), check_same_thread=False)
        connection.row_factory = sqlite3.Row
        return connection

    def _create_tables(self) -> None:
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS artifacts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_id TEXT UNIQUE NOT NULL,
                filename TEXT NOT NULL,
                source TEXT,
                pages INTEGER,
                original_size INTEGER,
                artifact_v1_path TEXT,
                artifact_v1_size INTEGER,
                artifact_v2_path TEXT,
                artifact_v2_size INTEGER,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL,
                metadata TEXT
            )
            """
        )
        self.connection.commit()

    def add_artifact(
        self,
        file_id: str,
        filename: str,
        source: Optional[str],
        pages: int,
        original_size: int,
        artifact_v1_path: str,
        artifact_v1_size: int,
        artifact_v2_path: Optional[str] = None,
        artifact_v2_size: Optional[int] = None,
        status: str = "ready",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.connection.execute(
            """
            INSERT OR REPLACE INTO artifacts (
                file_id,
                filename,
                source,
                pages,
                original_size,
                artifact_v1_path,
                artifact_v1_size,
                artifact_v2_path,
                artifact_v2_size,
                status,
                created_at,
                metadata
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                file_id,
                filename,
                source,
                pages,
                original_size,
                artifact_v1_path,
                artifact_v1_size,
                artifact_v2_path,
                artifact_v2_size,
                status,
                datetime.utcnow().isoformat() + "Z",
                json.dumps(metadata or {}),
            ),
        )
        self.connection.commit()

    def get_artifact(self, file_id: str) -> Optional[Dict[str, Any]]:
        cursor = self.connection.execute(
            "SELECT * FROM artifacts WHERE file_id = ?",
            (file_id,),
        )
        row = cursor.fetchone()
        if row is None:
            return None

        artifact = dict(row)
        artifact["metadata"] = json.loads(artifact["metadata"] or "{}")
        return artifact

    def list_artifacts(self) -> List[Dict[str, Any]]:
        cursor = self.connection.execute(
            "SELECT * FROM artifacts ORDER BY created_at DESC"
        )
        results = []
        for row in cursor.fetchall():
            item = dict(row)
            item["metadata"] = json.loads(item["metadata"] or "{}")
            results.append(item)
        return results

    def delete_artifact(self, file_id: str) -> None:
        self.connection.execute(
            "DELETE FROM artifacts WHERE file_id = ?",
            (file_id,),
        )
        self.connection.commit()
