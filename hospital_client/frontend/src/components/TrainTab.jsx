import "../App.css";
import { useState, useRef } from "react";
import PrivacySlider from "./PrivacySlider";
import CSVPreview from "./CSVPreview";
import { parseCSV, validateColumns, EXPECTED_COLUMNS } from "../utils/csv";

export default function TrainTab() {
  const [file, setFile] = useState(null);
  const [parsed, setParsed] = useState(null);
  const [error, setError] = useState("");
  const [privacyLevel, setPrivacyLevel] = useState(1.0);
  const [lastEpsilon, setLastEpsilon] = useState(null);
  const [status, setStatus] = useState(null); // null | 'loading' | 'success' | 'error'
  const [statusMsg, setStatusMsg] = useState("");
  const fileRef = useRef();

  function handleFile(f) {
    if (!f) return;
    if (!f.name.endsWith(".csv")) {
      setError("Please upload a valid .csv file.");
      setFile(null);
      setParsed(null);
      return;
    }
    const reader = new FileReader();
    reader.onload = (e) => {
      const text = e.target.result;
      const { headers, rows } = parseCSV(text);
      const { missing, extra } = validateColumns(headers, EXPECTED_COLUMNS);
      if (missing.length > 0) {
        setError(`Missing columns: ${missing.join(", ")}.x`);
        setParsed(null);
        setFile(null);
        return;
      } else if (extra.length > 0) {
        setError(`Extra columns: ${extra.join(", ")}.`);
        setParsed(null);
        setFile(null);
        return;
      }
      setError("");
      setFile(f);
      setParsed({ headers, rows });
    };
    reader.readAsText(f);
  }

  function handleDrop(e) {
    e.preventDefault();
    const f = e.dataTransfer.files[0];
    handleFile(f);
  }

  async function handleSubmit() {
    if (!file) return;
    setStatus("loading");
    setStatusMsg("");
    const epsilon = privacyLevel.toFixed(2);
    const formData = new FormData();
    formData.append("file", file);
    formData.append("epsilon", epsilon);
    try {
      const res = await fetch("/api/train", { method: "POST", body: formData });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || `Server responded with ${res.status}`);
      }
      const data = await res.json();
      setLastEpsilon(epsilon);
      setStatus("success");
      setStatusMsg(
        data.message ||
          "Model trained and weights uploaded to the central server successfully.",
      );
    } catch (err) {
      setStatus("error");
      setStatusMsg(err.message || "Failed to connect to backend.");
    }
  }

  function reset() {
    setFile(null);
    setParsed(null);
    setError("");
    setStatus(null);
    setStatusMsg("");
    setLastEpsilon(null);
    if (fileRef.current) fileRef.current.value = "";
  }

  function exportWeights() {
    const a = document.createElement("a");
    a.href = "/api/weights/export";
    a.download = "carp_dp_weights.pt";
    a.click();
  }

  return (
    <div className="tab-content">
      <div className="section-header">
        <h2>Upload Patient Dataset</h2>
        <p className="section-sub">
          Upload a CSV file with patient records to train the local model.
          Weights will trained locally for downloading.
        </p>
      </div>

      <div className="card">
        <label className="card-label">Dataset File</label>
        <div
          className={`dropzone ${file ? "dropzone-active" : ""}`}
          onDragOver={(e) => e.preventDefault()}
          onDrop={handleDrop}
          onClick={() => fileRef.current.click()}
        >
          <input
            ref={fileRef}
            type="file"
            accept=".csv"
            style={{ display: "none" }}
            onChange={(e) => handleFile(e.target.files[0])}
          />
          {file ? (
            <div className="dropzone-file">
              <span className="file-icon">📄</span>
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
                Drag & drop your CSV here, or{" "}
                <span className="link">browse</span>
              </p>
              <p className="dropzone-hint">
                Expected columns: {EXPECTED_COLUMNS.join(", ")}
              </p>
            </div>
          )}
        </div>
        {error && <p className="error-msg">{error}</p>}
      </div>

      {parsed && <CSVPreview headers={parsed.headers} rows={parsed.rows} />}

      <div className="card">
        <label className="card-label">Differential Privacy Settings</label>
        <PrivacySlider value={privacyLevel} onChange={setPrivacyLevel} />
      </div>

      {status === "success" && (
        <div className="alert alert-success">
          <div>
            <strong>Training Complete</strong>
            <p>{statusMsg}</p>
          </div>
        </div>
      )}
      {status === "error" && (
        <div className="alert alert-error">
          <span>❌</span>
          <div>
            <strong>Training Failed</strong>
            <p>{statusMsg}</p>
          </div>
        </div>
      )}

      <div className="action-row">
        <button
          className="btn btn-primary"
          disabled={!file || status === "loading"}
          onClick={handleSubmit}
        >
          {status === "loading" ? (
            <>
              <span className="spinner" /> Training Model...
            </>
          ) : (
            "Train Weights"
          )}
        </button>
        {status === "success" &&
          lastEpsilon &&
          parseFloat(lastEpsilon) < 9.5 && (
            <button
              className="btn btn-ghost"
              onClick={() => exportWeights()}
              title="Download differentially private model weights"
            >
              Export DP Weights
            </button>
          )}
        {file && status !== "loading" && (
          <button className="btn btn-ghost" onClick={reset}>
            Clear
          </button>
        )}
      </div>
    </div>
  );
}
