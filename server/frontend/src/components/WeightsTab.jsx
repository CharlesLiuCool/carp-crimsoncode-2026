import "../App.css";
import { useState, useRef } from "react";

const ACCEPTED_EXTENSIONS = [".pt", ".pth", ".pkl"];

function getFileExt(name) {
  return name.slice(name.lastIndexOf(".")).toLowerCase();
}

export default function WeightsTab() {
  const [file, setFile] = useState(null);
  const [error, setError] = useState("");
  const [status, setStatus] = useState(null); // null | 'loading' | 'success' | 'error'
  const [statusMsg, setStatusMsg] = useState("");
  const fileRef = useRef();

  function handleFile(f) {
    if (!f) return;
    const ext = getFileExt(f.name);
    if (!ACCEPTED_EXTENSIONS.includes(ext)) {
      setError(
        `Unsupported file type "${ext}". Please upload a .pt, .pth, or .pkl file.`,
      );
      setFile(null);
      return;
    }
    setError("");
    setStatus(null);
    setStatusMsg("");
    setFile(f);
  }

  function handleDrop(e) {
    e.preventDefault();
    handleFile(e.dataTransfer.files[0]);
  }

  function reset() {
    setFile(null);
    setError("");
    setStatus(null);
    setStatusMsg("");
    if (fileRef.current) fileRef.current.value = "";
  }

  async function handleUpload() {
    if (!file) return;
    setStatus("loading");
    setStatusMsg("");

    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await fetch("/api/weights/upload", {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || `Server responded with ${res.status}`);
      }

      const data = await res.json();
      setStatus("success");
      setStatusMsg(
        data.message || "Weights uploaded and registered successfully.",
      );
    } catch (err) {
      setStatus("error");
      setStatusMsg(err.message || "Failed to connect to the server.");
    }
  }

  return (
    <div className="tab-content">
      <div className="section-header">
        <h2>Upload Model Weights</h2>
        <p className="section-sub">
          Upload locally trained model weights to the central server. Accepted
          formats: <code>.pt</code>, <code>.pth</code>, <code>.pkl</code>.
        </p>
      </div>

      <div className="card">
        <label className="card-label">Weight File</label>
        <div
          className={`dropzone ${file ? "dropzone-active" : ""}`}
          onDragOver={(e) => e.preventDefault()}
          onDrop={handleDrop}
          onClick={() => fileRef.current.click()}
        >
          <input
            ref={fileRef}
            type="file"
            accept=".pt,.pth,.pkl"
            style={{ display: "none" }}
            onChange={(e) => handleFile(e.target.files[0])}
          />

          {file ? (
            <div className="dropzone-file">
              <div>
                <p className="file-name">{file.name}</p>
                <p className="file-size">{(file.size / 1024).toFixed(1)} KB</p>
              </div>
              <button
                className="remove-btn"
                onClick={(e) => {
                  e.stopPropagation();
                  reset();
                }}
              >
                ✕
              </button>
            </div>
          ) : (
            <div className="dropzone-empty">
              <p className="dropzone-text">
                Drag & drop your weights file here, or{" "}
                <span className="link">browse</span>
              </p>
              <p className="dropzone-hint">
                Accepted: {ACCEPTED_EXTENSIONS.join(", ")}
              </p>
            </div>
          )}
        </div>
        {error && <p className="error-msg">{error}</p>}
      </div>

      {status === "success" && (
        <div className="alert alert-success">
          <div>
            <strong>Upload Successful</strong>
            <p>{statusMsg}</p>
          </div>
        </div>
      )}

      {status === "error" && (
        <div className="alert alert-error">
          <div>
            <strong>Upload Failed</strong>
            <p>{statusMsg}</p>
          </div>
        </div>
      )}

      <div className="action-row">
        <button
          className="btn btn-primary"
          disabled={!file || status === "loading"}
          onClick={handleUpload}
        >
          {status === "loading" ? (
            <>
              <span className="spinner" /> Uploading...
            </>
          ) : (
            "Upload Weights"
          )}
        </button>
        {file && status !== "loading" && (
          <button className="btn btn-ghost" onClick={reset}>
            Clear
          </button>
        )}
      </div>
    </div>
  );
}
