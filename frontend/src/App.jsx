import React, { useState, useEffect } from 'react';
import { Upload, Loader2, ChevronRight, CheckCircle2, AlertCircle, BookOpen, Clock, Download, Archive } from 'lucide-react';

const API_BASE = "http://localhost:8000";
const SESSION_STORAGE_KEY = "liteRAG-session";
const EMPTY_SESSION = { id: null, documentName: null, originalSize: null, artifactSize: null };

const loadStoredSession = () => {
  try {
    const raw = window.localStorage.getItem(SESSION_STORAGE_KEY);
    return raw ? { ...EMPTY_SESSION, ...JSON.parse(raw) } : EMPTY_SESSION;
  } catch {
    return EMPTY_SESSION;
  }
};

const saveStoredSession = (session) => {
  try {
    window.localStorage.setItem(SESSION_STORAGE_KEY, JSON.stringify(session));
  } catch {
    // Ignore storage failures and continue with in-memory state.
  }
};

const clearStoredSession = () => {
  try {
    window.localStorage.removeItem(SESSION_STORAGE_KEY);
  } catch {
    // Ignore storage failures and continue with in-memory state.
  }
};

const formatBytes = (bytes) => {
  if (!bytes || bytes <= 0) return '0 B';
  const units = ['B', 'KB', 'MB', 'GB'];
  const order = Math.min(Math.floor(Math.log(bytes) / Math.log(1024)), units.length - 1);
  const value = bytes / 1024 ** order;
  const precision = order === 0 ? 0 : order === 1 ? 1 : 2;
  return `${value.toFixed(precision)} ${units[order]}`;
};

const buildArtifactFilename = (fileId) => `artifact_${fileId || 'document'}.json`;

const MemoizedSpinner = React.memo(({ size = 24, color = "currentColor", className = "" }) => (
  <Loader2 size={size} color={color} className={`animate-spin ${className}`} style={{ willChange: 'transform' }} />
));

// --- Helpers ---
const ArtifactPanel = ({ session, artifactState, onDownload }) => {
  const originalSize = artifactState.originalSize ?? session.originalSize ?? 0;
  const compressedSize = artifactState.size ?? session.artifactSize ?? 0;
  const ratio = originalSize > 0 && compressedSize > 0 ? originalSize / compressedSize : null;
  const compressedPercent = originalSize > 0 && compressedSize > 0
    ? Math.max(10, Math.min(100, (compressedSize / originalSize) * 100))
    : 18;

  return (
    <section className="artifact-panel animate-slide-up">
      <div className="artifact-panel__header">
        <div>
          <p className="mono-text artifact-panel__eyebrow">Compression Result</p>
          <h2 className="artifact-panel__title">Knowledge artifact ready for export.</h2>
          <p className="artifact-panel__subtitle">
            The distilled JSON is prepared for reuse, inspection, and downstream retrieval workflows.
          </p>
        </div>

        <button
          type="button"
          className="artifact-download-btn"
          onClick={onDownload}
          disabled={artifactState.status !== 'ready'}
        >
          {artifactState.status === 'loading' ? (
            <>
              <Loader2 size={16} className="animate-spin" />
              Preparing Artifact...
            </>
          ) : (
            <>
              <Download size={16} />
              Download Artifact
            </>
          )}
        </button>
      </div>

      <div className="compression-card">
        <div className="compression-card__meta">
          <div className="compression-stat">
            <span className="compression-stat__label">Original Size</span>
            <strong className="compression-stat__value">{originalSize ? formatBytes(originalSize) : 'Waiting...'}</strong>
          </div>
          <div className="compression-stat">
            <span className="compression-stat__label">Compressed Size</span>
            <strong className="compression-stat__value">{compressedSize ? formatBytes(compressedSize) : 'Preparing...'}</strong>
          </div>
          <div className="compression-stat">
            <span className="compression-stat__label">Reduction</span>
            <strong className="compression-stat__value compression-stat__value--accent">
              {ratio ? `${ratio.toFixed(1)}x smaller` : 'Calculating...'}
            </strong>
          </div>
        </div>

        <div className="compression-bars">
          <div className="compression-bar-row">
            <div className="compression-bar-row__header">
              <span className="compression-bar-row__label">Original</span>
              <span className="compression-bar-row__size">{originalSize ? formatBytes(originalSize) : '--'}</span>
            </div>
            <div className="compression-bar-track">
              <div className="compression-bar compression-bar--original" style={{ width: '100%' }} />
            </div>
          </div>

          <div className="compression-bar-row">
            <div className="compression-bar-row__header">
              <span className="compression-bar-row__label">Compressed</span>
              <span className="compression-bar-row__size">{compressedSize ? formatBytes(compressedSize) : '--'}</span>
            </div>
            <div className="compression-bar-track">
              <div className="compression-bar compression-bar--compressed" style={{ width: `${compressedPercent}%` }} />
            </div>
          </div>
        </div>

        <div className="compression-card__footer">
          <span className="compression-card__note">
            <Archive size={14} />
            JSON artifact keeps distilled structure while trimming bulk from the source document.
          </span>
          <span className="compression-card__filename">{buildArtifactFilename(session.id)}</span>
        </div>
      </div>
    </section>
  );
};

// --- Main App ---

function App() {
  const [session, setSession] = useState(() => loadStoredSession());
  const [artifactState, setArtifactState] = useState({ status: 'idle', size: null, originalSize: null, fetchedFor: null });
  const [artifacts, setArtifacts] = useState([]);
  const [status, setStatus] = useState('checking'); // checking, idle, uploading, ready
  const [uploadStages, setUploadStages] = useState([]);
  const [isDragging, setIsDragging] = useState(false);
  const [isResetting, setIsResetting] = useState(false);

  const handleDragOver = (e) => {
    e.preventDefault();
    setIsDragging(true);
  };
  const handleDragLeave = (e) => {
    e.preventDefault();
    setIsDragging(false);
  };
  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleUpload({ target: { files: [e.dataTransfer.files[0]] } });
    }
  };
  const syncSession = (nextSession) => {
    setSession(nextSession);
    saveStoredSession(nextSession);
  };

  const loadArtifacts = async () => {
    try {
      const response = await fetch(`${API_BASE}/artifacts`);
      if (!response.ok) throw new Error('Failed to fetch artifacts');
      const data = await response.json();
      setArtifacts(Array.isArray(data.artifacts) ? data.artifacts : []);
    } catch (err) {
      console.error(err);
    }
  };

  useEffect(() => {
    if (status !== 'ready' || !session.id) return;

    let cancelled = false;
    let intervalId;

    const pollArtifactStatus = async () => {
      if (!cancelled) {
        setArtifactState((prev) => ({
          status: prev.status === 'ready' && prev.fetchedFor === session.id ? 'ready' : 'loading',
          size: prev.size,
          originalSize: prev.originalSize,
          fetchedFor: session.id,
        }));
      }

      try {
        const response = await fetch(`${API_BASE}/artifact/status?file_id=${session.id}`);
        if (!response.ok) throw new Error('Artifact status unavailable');

        const data = await response.json();
        console.log('/artifact/status', data);
        if (cancelled) return;

        if (data.ready && data.artifact) {
          const artifact = data.artifact;
          const resolvedSession = {
            ...session,
            id: artifact.file_id || session.id,
            originalSize: artifact.original_size || session.originalSize || null,
            artifactSize: artifact.artifact_v2_size || artifact.artifact_v1_size || null,
          };

          setArtifactState({
            status: 'ready',
            size: artifact.artifact_v2_size || artifact.artifact_v1_size || null,
            originalSize: artifact.original_size || null,
            fetchedFor: resolvedSession.id,
          });
          setSession(resolvedSession);
          saveStoredSession(resolvedSession);
          return;
        }

        setArtifactState({
          status: 'loading',
          size: null,
          originalSize: session.originalSize || null,
          fetchedFor: session.id,
        });
      } catch (err) {
        if (cancelled) return;
        console.error(err);
        setArtifactState({ status: 'error', size: null, originalSize: session.originalSize || null, fetchedFor: session.id });
      }
    };

    pollArtifactStatus();
    intervalId = window.setInterval(pollArtifactStatus, 1500);

    return () => {
      cancelled = true;
      window.clearInterval(intervalId);
    };
  }, [status, session.id]);

  useEffect(() => {
    let cancelled = false;

    const initializeFromBackend = async () => {
      try {
        const response = await fetch(`${API_BASE}/status`);
        if (!response.ok) throw new Error('Status check failed');

        const data = await response.json();
        if (cancelled) return;

        if (data.indexed) {
          const storedSession = loadStoredSession();
          const nextSession = storedSession?.documentName
            ? storedSession
            : { ...EMPTY_SESSION, id: 'persisted-index', documentName: data.source || 'Indexed document' };

          setSession(nextSession);
          saveStoredSession(nextSession);
          setStatus('ready');
          return;
        }

        clearStoredSession();
        setSession(EMPTY_SESSION);
        setArtifactState({ status: 'idle', size: null, originalSize: null, fetchedFor: null });
        setStatus('idle');
      } catch (err) {
        if (cancelled) return;
        console.error(err);
        setStatus('idle');
      }
    };

    initializeFromBackend();
    loadArtifacts();

    return () => {
      cancelled = true;
    };
  }, []);

  const addUploadStage = (msg, isCurrent = false, isDone = false) => {
    setUploadStages(prev => [...prev.filter(s => s.msg !== msg), { msg, isCurrent, isDone }]);
  };

  const handleUpload = async (e) => {
    const uploadedFile = e.target.files[0];
    if (!uploadedFile) return;

    setStatus('uploading');
    setUploadStages([]);
    setArtifactState({ status: 'loading', size: null, originalSize: uploadedFile.size, fetchedFor: null });

    // UI Simulation for UX Feedback
    addUploadStage('Extracting...', true, false);

    const formData = new FormData();
    formData.append('file', uploadedFile);

    try {
      setTimeout(() => addUploadStage('Extracting...', false, true), 300);
      setTimeout(() => addUploadStage('Chunking...', true, false), 400);

      const response = await fetch(`${API_BASE}/upload`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) throw new Error('Upload failed');
      const data = await response.json();

      addUploadStage('Chunking...', false, true);
      addUploadStage('Embedding...', false, true);
      addUploadStage('Indexing...', true, false);

      setTimeout(() => {
        addUploadStage('Indexing...', false, true);
        const nextSession = {
          ...EMPTY_SESSION,
          id: data.file_id || Date.now(),
          documentName: uploadedFile.name,
          originalSize: uploadedFile.size,
          artifactSize: data.artifact_v2_size || data.artifact_v1_size || null,
        };
        syncSession(nextSession);
        setArtifactState({
          status: 'ready',
          size: data.artifact_v2_size || data.artifact_v1_size || null,
          originalSize: uploadedFile.size,
          fetchedFor: nextSession.id,
        });
        setStatus('ready');
        loadArtifacts();
      }, 500);

    } catch (err) {
      console.error(err);
      setStatus('idle');
      alert('Failed to upload document. Please ensure backend is running.');
    }
  };

  const handleResetSession = async () => {
    if (isResetting) return;

    const confirmed = window.confirm(
      'Start a new session? This will clear the currently indexed document and cached answers.'
    );

    if (!confirmed) return;

    setIsResetting(true);

    try {
      const response = await fetch(`${API_BASE}/reset`, {
        method: 'POST',
      });

      if (!response.ok) throw new Error('Reset failed');

      clearStoredSession();
      setSession(EMPTY_SESSION);
      setArtifactState({ status: 'idle', size: null, originalSize: null, fetchedFor: null });
      setUploadStages([]);
      setStatus('idle');
    } catch (err) {
      console.error(err);
      alert('Failed to reset the current session. Please try again.');
    } finally {
      setIsResetting(false);
    }
  };

  const handleDownloadArtifact = async () => {
    if (artifactState.status !== 'ready' || !session.id) return;
    await downloadArtifactById(session.id, 'v2');
  };

  const downloadArtifactById = async (fileId, version = 'v2') => {
    if (!fileId) return;

    try {
      const response = await fetch(`${API_BASE}/artifact/download/${fileId}?version=${version}`);
      if (!response.ok) throw new Error('Artifact download unavailable');

      const blob = await response.blob();
      const contentDisposition = response.headers.get('Content-Disposition');
      let filename = buildArtifactFilename(fileId);

      if (contentDisposition && contentDisposition.includes('filename=')) {
        filename = contentDisposition.split('filename=')[1].replace(/"/g, '');
      }

      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = filename;
      document.body.appendChild(a);
      a.click();
      a.remove();
      window.URL.revokeObjectURL(url);
    } catch (err) {
      console.error(err);
      alert('Failed to download the knowledge artifact. Please try again.');
    }
  };


  return (
    <div className="container" style={{ position: 'relative', paddingBottom: '8rem' }}>

      {/* Editorial Header */}
      <header className="animate-fade-in" style={{ marginBottom: '4rem', display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
        <div>
          <h1 style={{ marginBottom: '0.25rem' }}>liteRAG.</h1>
          <p className="mono-text" style={{ color: 'var(--text-muted)' }}>RESEARCH ENGINE PIPELINE</p>
        </div>

        {/* Session Context */}
        {session.documentName && (
          <div style={{ textAlign: 'right', display: 'flex', flexDirection: 'column', alignItems: 'flex-end', gap: '0.25rem' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', color: 'var(--text-secondary)' }}>
              <BookOpen size={14} />
              <span className="mono-text">{session.documentName}</span>
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', color: 'var(--text-muted)' }}>
              <Clock size={12} />
              <span className="mono-text" style={{ fontSize: '0.7rem' }}>Session Active</span>
            </div>
            {status === 'ready' && (
              <button
                type="button"
                className="session-reset-btn"
                onClick={handleResetSession}
                disabled={isResetting}
              >
                {isResetting ? 'Resetting...' : 'Start New Session'}
              </button>
            )}
          </div>
        )}
      </header>

      {status === 'checking' && (
        <div className="animate-fade-in" style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', minHeight: '50vh', gap: '1rem', textAlign: 'center' }}>
          <MemoizedSpinner size={28} color="var(--accent)" />
          <div>
            <h2 style={{ marginBottom: '0.5rem' }}>Checking Indexed State</h2>
            <p style={{ color: 'var(--text-secondary)', maxWidth: '420px' }}>
              Reconnecting the interface to the persisted backend index so the correct workspace loads after refresh.
            </p>
          </div>
        </div>
      )}

      {/* Upload State */}
      {(status === 'idle' || status === 'uploading') && (
        <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', width: '100%' }}>
          <div
            className={`file-dropzone animate-fade-in ${status === 'uploading' ? 'pulse-border' : ''} ${isDragging ? 'dragging' : ''}`}
            style={{
              padding: '4rem 3rem',
              borderRadius: 'var(--radius-sm)',
              width: '100%',
              maxWidth: '600px',
              borderColor: isDragging ? 'var(--accent)' : ''
            }}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
          >
            <input type="file" id="pdf-upload" accept=".pdf" onChange={handleUpload} style={{ display: 'none' }} disabled={status === 'uploading'} />

            <label htmlFor="pdf-upload" style={{ cursor: status === 'uploading' ? 'default' : 'pointer', display: 'block', width: '100%', margin: 0 }}>
              {status === 'idle' ? (
                <div style={{ textAlign: 'center' }}>
                  <div style={{ marginBottom: '1.5rem', opacity: isDragging ? 1 : 0.8, color: isDragging ? 'var(--accent)' : 'var(--text-primary)', transition: 'all 0.2s ease' }}>
                    <Upload size={48} strokeWidth={1.5} color="currentColor" />
                  </div>
                  <h2 style={{ marginBottom: '1rem', color: isDragging ? 'var(--accent)' : 'var(--text-primary)', transition: 'color 0.2s ease' }}>
                    {isDragging ? 'Drop your PDF here' : 'Upload PDF'}
                  </h2>
                  <p style={{ color: 'var(--text-secondary)', maxWidth: '400px', margin: '0 auto', fontFamily: 'var(--font-body)', lineHeight: 1.6 }}>
                    Transform complex documents into structured synthesis through privacy-first semantic search and contextual reasoning.
                  </p>
                  <div style={{ marginTop: '2.5rem' }}>
                    <span className="btn-primary">Select PDF to Analyze</span>
                  </div>
                </div>
              ) : (
                <div style={{ maxWidth: '400px', margin: '0 auto' }}>
                  <h2 style={{ marginBottom: '2rem', textAlign: 'center' }}>Processing...</h2>
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
                    {uploadStages.map((stage, i) => (
                      <div key={i} className="animate-fade-in" style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
                        {stage.isDone ? (
                          <CheckCircle2 size={18} color="var(--success)" />
                        ) : stage.isCurrent ? (
                          <MemoizedSpinner size={18} color="var(--accent)" />
                        ) : (
                          <div style={{ width: '18px', height: '18px', border: '1px solid var(--border)', borderRadius: '50%' }} />
                        )}
                        <span className="mono-text" style={{
                          color: stage.isCurrent ? 'var(--text-primary)' : (stage.isDone ? 'var(--text-secondary)' : 'var(--text-muted)'),
                          opacity: stage.isDone ? 0.7 : 1
                        }}>
                          {stage.msg}
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </label>
          </div>

          {(status === 'idle' || status === 'uploading') && (
            <div className="animate-fade-in" style={{ marginTop: '4rem', display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '2rem' }}>
              <div className="system-pipeline" style={{ display: 'flex', alignItems: 'center', gap: '1rem', fontSize: '0.85rem', fontFamily: 'var(--font-mono)', letterSpacing: '0.05em', fontWeight: 500 }}>
                <span className={`pipeline-step ${uploadStages.find(s => s.msg === 'Extracting...' && s.isCurrent) ? 'active' : ''} ${uploadStages.find(s => s.msg === 'Extracting...' && s.isDone) ? 'done' : ''}`}>EXTRACT</span>
                <ChevronRight size={14} className="pipeline-chevron" />
                <span className={`pipeline-step ${uploadStages.find(s => s.msg === 'Chunking...' && s.isCurrent) ? 'active' : ''} ${uploadStages.find(s => s.msg === 'Chunking...' && s.isDone) ? 'done' : ''}`}>CHUNK</span>
                <ChevronRight size={14} className="pipeline-chevron" />
                <span className={`pipeline-step ${uploadStages.find(s => (s.msg === 'Embedding...' || s.msg === 'Indexing...') && s.isCurrent) ? 'active' : ''} ${uploadStages.find(s => s.msg === 'Indexing...' && s.isDone) ? 'done' : ''}`}>EMBED</span>
                <ChevronRight size={14} className="pipeline-chevron" />
                <span className="pipeline-step">EXPORT</span>
              </div>

              <div style={{ display: 'flex', alignItems: 'center', gap: '2rem', color: 'var(--text-muted)', fontSize: '0.85rem' }}>
                <span style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                  <CheckCircle2 size={16} color="var(--success)" /> Processed locally
                </span>
                <span style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                  <CheckCircle2 size={16} color="var(--success)" /> No external storage
                </span>
              </div>
            </div>
          )}
        </div>
      )}

      {/* Research Transcript State */}
      {status === 'ready' && (
        <div className="transcript-container animate-fade-in">
          {session.documentName && (
            <ArtifactPanel
              session={session}
              artifactState={artifactState}
              onDownload={handleDownloadArtifact}
            />
          )}

          {artifacts.length > 0 && (
            <section className="artifact-list-panel" style={{ marginTop: '2rem', padding: '1.75rem', borderRadius: 'var(--radius-sm)', background: 'var(--bg-secondary)', border: '1px solid var(--border)' }}>
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: '1rem', marginBottom: '1rem' }}>
                <div>
                  <p className="mono-text" style={{ marginBottom: '0.5rem', color: 'var(--text-secondary)', fontSize: '0.75rem', letterSpacing: '0.08em' }}>Stored artifacts</p>
                  <h3 style={{ margin: 0, fontSize: '1.15rem', color: 'var(--text-primary)' }}>{artifacts.length} artifact{artifacts.length === 1 ? '' : 's'} available</h3>
                </div>
                <button
                  type="button"
                  className="btn-secondary"
                  onClick={loadArtifacts}
                  style={{ padding: '0.75rem 1rem', borderRadius: '999px', border: '1px solid var(--border)', background: 'transparent', color: 'var(--text-primary)', cursor: 'pointer' }}
                >
                  Refresh list
                </button>
              </div>
              <div style={{ display: 'grid', gap: '1rem' }}>
                {artifacts.map((artifact) => (
                  <div key={artifact.file_id} style={{ display: 'grid', gap: '0.75rem', padding: '1rem', background: 'var(--bg-primary)', borderRadius: 'var(--radius-sm)', border: '1px solid var(--border)' }}>
                    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: '1rem' }}>
                      <div>
                        <strong style={{ fontSize: '0.95rem', color: 'var(--text-primary)' }}>{artifact.filename || artifact.file_id}</strong>
                        <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.75rem', marginTop: '0.25rem', color: 'var(--text-muted)', fontSize: '0.82rem' }}>
                          <span>{artifact.pages ? `${artifact.pages} pages` : 'Page count unknown'}</span>
                          <span>{artifact.artifact_v2_size ? formatBytes(artifact.artifact_v2_size) : formatBytes(artifact.artifact_v1_size)}</span>
                        </div>
                      </div>
                      <button
                        type="button"
                        className="btn-secondary"
                        onClick={() => downloadArtifactById(artifact.file_id)}
                        style={{ padding: '0.75rem 1rem', borderRadius: '999px', border: '1px solid var(--border)', background: 'transparent', color: 'var(--text-primary)', cursor: 'pointer' }}
                      >
                        Download
                      </button>
                    </div>
                    <div style={{ display: 'flex', flexWrap: 'wrap', gap: '1rem', color: 'var(--text-muted)', fontSize: '0.82rem' }}>
                      <span>Original: {artifact.original_size ? formatBytes(artifact.original_size) : 'N/A'}</span>
                      <span>Compressed: {artifact.artifact_v2_size ? formatBytes(artifact.artifact_v2_size) : formatBytes(artifact.artifact_v1_size)}</span>
                    </div>
                  </div>
                ))}
              </div>
            </section>
          )}

              <div style={{ padding: '4rem 0', opacity: 0.5, maxWidth: '600px' }}>
            <h2 style={{ marginBottom: '1rem' }}>Document Indexed.</h2>
            <p style={{ fontSize: '1.125rem' }}>Your compressed artifact and metadata are ready for download and reuse.</p>
          </div>
        </div>
      )}


    </div>
  );
}

export default App;
