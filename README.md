# liteRAG

A production-ready PDF artifact generator that converts large PDF documents into compressed, AI-friendly JSON artifacts and persists them in a local SQLite registry.

## 🛠️ Tech Stack

- **Backend**: FastAPI, PyMuPDF, FAISS, sentence-transformers, Google GenAI (Gemini-3-Flash).
- **Frontend**: Vite, React, Lucide React, CSS Variables.

## 🚀 Quick Start

### Backend

1. Activate virtual environment:
   - **PowerShell**: `.\.venv\Scripts\Activate.ps1`
   - **Bash**: `source .venv/bin/activate`
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Set your Google API Key:
   - **PowerShell**: `$env:GOOGLE_API_KEY='your-api-key'`
   - **Bash**: `export GOOGLE_API_KEY='your-api-key'`
   - **Alternative**: Create a `.env` file in the `backend` directory with `GOOGLE_API_KEY=your-api-key`.
3. Run the API:
   ```bash
   uvicorn app.main:app --reload
   ```

### Artifact Endpoints

Once the backend is running, use these endpoints to upload and manage compressed PDF artifacts:

- `POST /upload` — upload a PDF and generate compressed artifact JSON files.
- `GET /artifacts` — list stored artifact records from the SQLite artifact registry.
- `GET /artifact/{file_id}` — get metadata for a specific artifact.
- `GET /artifact/download/{file_id}?version=v2` — download the compressed v2 artifact JSON.
- `GET /artifact/status?file_id={file_id}` — check artifact readiness and metadata.

> No manual database setup is required for the default local artifact registry. The backend creates `backend/data/artifact_store.db` automatically when it first runs.
>
> To switch to a SQL database later, set these env vars in `backend/.env`:
> - `ARTIFACT_STORE_TYPE=sqlalchemy`
> - `ARTIFACT_STORE_DSN=your-database-dsn`
> - `ARTIFACT_STORAGE_DIR=backend/data/artifacts`
>
> For local SQLite only, you can also override `ARTIFACT_STORE_PATH`.

### Frontend

1. Install dependencies:
   ```bash
   cd frontend
   npm install
   ```
2. Run the dev server:
   ```bash
   npm run dev
   ```

## 🔍 Evaluation

To run the retrieval validation suite:
```bash
python -m backend.tests.run_validation
```

## ⚖️ License
MIT
