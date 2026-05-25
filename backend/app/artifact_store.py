import json
import os
import sqlite3
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

try:
    from sqlalchemy import Column, Integer, String, Text, create_engine
    from sqlalchemy.ext.declarative import declarative_base
    from sqlalchemy.orm import Session, sessionmaker
    SQLALCHEMY_AVAILABLE = True
except ImportError:
    SQLALCHEMY_AVAILABLE = False

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DEFAULT_DB_PATH = DATA_DIR / "artifact_store.db"
DEFAULT_ARTIFACT_DIR = DATA_DIR / "artifacts"

load_dotenv(BASE_DIR / ".env")

ARTIFACT_STORE_TYPE = os.getenv("ARTIFACT_STORE_TYPE", "sqlite").strip().lower()
ARTIFACT_STORE_PATH = Path(os.getenv("ARTIFACT_STORE_PATH", str(DEFAULT_DB_PATH)))
ARTIFACT_STORE_DSN = os.getenv("ARTIFACT_STORE_DSN", "").strip()
ARTIFACT_STORAGE_DIR = Path(os.getenv("ARTIFACT_STORAGE_DIR", str(DEFAULT_ARTIFACT_DIR)))
ARTIFACT_STORAGE_DIR.mkdir(parents=True, exist_ok=True)

Base = declarative_base()


class ArtifactStoreBase(ABC):
    @abstractmethod
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
        pass

    @abstractmethod
    def get_artifact(self, file_id: str) -> Optional[Dict[str, Any]]:
        pass

    @abstractmethod
    def list_artifacts(self) -> List[Dict[str, Any]]:
        pass

    @abstractmethod
    def delete_artifact(self, file_id: str) -> None:
        pass


class SQLiteArtifactStore(ArtifactStoreBase):
    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = Path(db_path or ARTIFACT_STORE_PATH)
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


if SQLALCHEMY_AVAILABLE:
    class ArtifactModel(Base):
        __tablename__ = "artifacts"

        id = Column(Integer, primary_key=True, autoincrement=True)
        file_id = Column(String, unique=True, nullable=False)
        filename = Column(String, nullable=False)
        source = Column(String)
        pages = Column(Integer)
        original_size = Column(Integer)
        artifact_v1_path = Column(String)
        artifact_v1_size = Column(Integer)
        artifact_v2_path = Column(String)
        artifact_v2_size = Column(Integer)
        status = Column(String, nullable=False)
        created_at = Column(String, nullable=False)
        metadata = Column(Text)


    class SQLAlchemyArtifactStore(ArtifactStoreBase):
        def __init__(self, dsn: str):
            if not dsn:
                raise ValueError("ARTIFACT_STORE_DSN must be provided for SQLAlchemy database storage.")
            self.engine = create_engine(dsn, future=True)
            self.SessionLocal = sessionmaker(bind=self.engine, expire_on_commit=False)
            Base.metadata.create_all(self.engine)

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
            with self.SessionLocal() as session:
                artifact = session.query(ArtifactModel).filter_by(file_id=file_id).one_or_none()
                if artifact is None:
                    artifact = ArtifactModel(file_id=file_id)
                artifact.filename = filename
                artifact.source = source
                artifact.pages = pages
                artifact.original_size = original_size
                artifact.artifact_v1_path = artifact_v1_path
                artifact.artifact_v1_size = artifact_v1_size
                artifact.artifact_v2_path = artifact_v2_path
                artifact.artifact_v2_size = artifact_v2_size
                artifact.status = status
                artifact.created_at = datetime.utcnow().isoformat() + "Z"
                artifact.metadata = json.dumps(metadata or {})
                session.add(artifact)
                session.commit()

        def get_artifact(self, file_id: str) -> Optional[Dict[str, Any]]:
            with self.SessionLocal() as session:
                artifact = session.query(ArtifactModel).filter_by(file_id=file_id).one_or_none()
                if artifact is None:
                    return None
                return {
                    "id": artifact.id,
                    "file_id": artifact.file_id,
                    "filename": artifact.filename,
                    "source": artifact.source,
                    "pages": artifact.pages,
                    "original_size": artifact.original_size,
                    "artifact_v1_path": artifact.artifact_v1_path,
                    "artifact_v1_size": artifact.artifact_v1_size,
                    "artifact_v2_path": artifact.artifact_v2_path,
                    "artifact_v2_size": artifact.artifact_v2_size,
                    "status": artifact.status,
                    "created_at": artifact.created_at,
                    "metadata": json.loads(artifact.metadata or "{}"),
                }

        def list_artifacts(self) -> List[Dict[str, Any]]:
            with self.SessionLocal() as session:
                results = []
                artifacts = session.query(ArtifactModel).order_by(ArtifactModel.created_at.desc()).all()
                for artifact in artifacts:
                    results.append({
                        "id": artifact.id,
                        "file_id": artifact.file_id,
                        "filename": artifact.filename,
                        "source": artifact.source,
                        "pages": artifact.pages,
                        "original_size": artifact.original_size,
                        "artifact_v1_path": artifact.artifact_v1_path,
                        "artifact_v1_size": artifact.artifact_v1_size,
                        "artifact_v2_path": artifact.artifact_v2_path,
                        "artifact_v2_size": artifact.artifact_v2_size,
                        "status": artifact.status,
                        "created_at": artifact.created_at,
                        "metadata": json.loads(artifact.metadata or "{}"),
                    })
                return results

        def delete_artifact(self, file_id: str) -> None:
            with self.SessionLocal() as session:
                session.query(ArtifactModel).filter_by(file_id=file_id).delete()
                session.commit()


def create_artifact_store() -> ArtifactStoreBase:
    if ARTIFACT_STORE_TYPE != "sqlite" or ARTIFACT_STORE_DSN:
        if not SQLALCHEMY_AVAILABLE:
            raise RuntimeError(
                "SQLAlchemy is required for non-SQLite artifact storage. Install sqlalchemy or set ARTIFACT_STORE_TYPE=sqlite."
            )
        return SQLAlchemyArtifactStore(ARTIFACT_STORE_DSN)

    return SQLiteArtifactStore(db_path=ARTIFACT_STORE_PATH)
