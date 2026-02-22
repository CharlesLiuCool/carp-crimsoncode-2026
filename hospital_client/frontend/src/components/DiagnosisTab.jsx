import "../App.css";
import { useState, useRef } from "react";
import { parseCSV, validateColumns, DIAGNOSIS_COLUMNS } from "../utils/csv";
import CSVPreview from "./CSVPreview";

const fieldMeta = {
  Age: {
    label: "Age",
    unit: "years",
    min: 1,
    max: 120,
    step: 1,
    placeholder: "45",
  },
  BMI: {
    label: "BMI",
    unit: "kg/m²",
    min: 10,
    max: 70,
    step: 0.1,
    placeholder: "28.5",
  },
  Glucose: {
    label: "Glucose",
    unit: "mg/dL",
    min: 0,
    max: 400,
    step: 1,
    placeholder: "120",
  },
};

export default function DiagnosisTab() {
  const [mode, setMode] = useState("form");
  const [formData, setFormData] = useState({ Age: "", BMI: "", Glucose: "" });
  const [file, setFile] = useState(null);
  const [parsed, setParsed] = useState(null);
  const [csvError, setCsvError] = useState("");
  const [status, setStatus] = useState(null); // null | 'loading' | 'done' | 'error'
  const [result, setResult] = useState(null);
  const fileRef = useRef();

  function handleFormChange(field, value) {
    setFormData((prev) => ({ ...prev, [field]: value }));
  }

  function handleFile(f) {
    if (!f || !f.name.endsWith(".csv")) {
      setCsvError("Please upload a valid .csv file.");
      return;
    }
    const reader = new FileReader();
    reader.onload = (e) => {
      const { headers, rows } = parseCSV(e.target.result);
      const { missing } = validateColumns(headers, DIAGNOSIS_COLUMNS);
      if (missing.length > 0) {
        setCsvError(`Missing columns: ${missing.join(", ")}`);
        return;
      }
      setCsvError("");
      setFile(f);
      setParsed({ headers, rows });
    };
    reader.readAsText(f);
  }

  const formComplete = DIAGNOSIS_COLUMNS.every((k) => formData[k] !== "");

  async function handleSubmit() {
    setStatus("loading");
    setResult(null);
    try {
      let res;
      if (mode === "form") {
        res = await fetch("/api/diagnose", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            Age: Number(formData.Age),
            BMI: Number(formData.BMI),
            Glucose: Number(formData.Glucose),
          }),
        });
      } else {
        const fd = new FormData();
        fd.append("file", file);
        res = await fetch("/api/diagnose/batch", { method: "POST", body: fd });
      }
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || `Server responded with ${res.status}`);
      }
      const data = await res.json();
      setStatus("done");
      setResult(data);
    } catch (err) {
      setStatus("error");
      setResult({ error: err.message });
    }
  }

  function reset() {
    setFile(null);
    setParsed(null);
    setCsvError("");
    setStatus(null);
    setResult(null);
    if (fileRef.current) fileRef.current.value = "";
  }

  // Single result (form mode)
  const singleResult =
    mode === "form" && status === "done" && result && !result.error
      ? result
      : null;

  // Batch results (CSV mode)
  const batchResults =
    mode === "csv" && status === "done" && result?.results
      ? result.results
      : null;

  return (
    <div className="tab-content">
      <div className="section-header">
        <h2>Patient Diagnosis</h2>
        <p className="section-sub">
          Submit patient data to receive a diabetes risk assessment from the
          trained model. Requires Age, BMI, and fasting Glucose.
        </p>
      </div>

      <div className="mode-toggle">
        <button
          className={`mode-btn ${mode === "form" ? "active" : ""}`}
          onClick={() => {
            setMode("form");
            reset();
          }}
        >
          Manual Entry
        </button>
        <button
          className={`mode-btn ${mode === "csv" ? "active" : ""}`}
          onClick={() => {
            setMode("csv");
            reset();
          }}
        >
          Upload CSV
        </button>
      </div>

      {/* ── Manual entry form ── */}
      {mode === "form" && (
        <div className="card">
          <label className="card-label">Patient Information</label>
          <div className="form-grid">
            {DIAGNOSIS_COLUMNS.map((field) => {
              const meta = fieldMeta[field];
              return (
                <div className="form-field" key={field}>
                  <label>{meta.label}</label>
                  <div className="input-wrapper">
                    <input
                      type="number"
                      min={meta.min}
                      max={meta.max}
                      step={meta.step}
                      placeholder={meta.placeholder}
                      value={formData[field]}
                      onChange={(e) => handleFormChange(field, e.target.value)}
                    />
                    {meta.unit && <span className="unit">{meta.unit}</span>}
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* ── CSV upload ── */}
      {mode === "csv" && (
        <div className="card">
          <label className="card-label">Patient CSV File</label>
          <div
            className={`dropzone ${file ? "dropzone-active" : ""}`}
            onDragOver={(e) => e.preventDefault()}
            onDrop={(e) => {
              e.preventDefault();
              handleFile(e.dataTransfer.files[0]);
            }}
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
                  <p className="file-size">
                    {(file.size / 1024).toFixed(1)} KB
                  </p>
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
                  Drag & drop CSV, or <span className="link">browse</span>
                </p>
                <p className="dropzone-hint">
                  Columns: {DIAGNOSIS_COLUMNS.join(", ")}
                </p>
              </div>
            )}
          </div>
          {csvError && <p className="error-msg">{csvError}</p>}
          {parsed && <CSVPreview headers={parsed.headers} rows={parsed.rows} />}
        </div>
      )}

      {/* ── Single result ── */}
      {singleResult && (
        <div
          className={`result-card ${singleResult.prediction === 1 ? "result-positive" : "result-negative"}`}
        >
          <div className="result-body">
            <h3>
              {singleResult.prediction === 1
                ? "Diabetes Risk Detected"
                : "No Diabetes Risk Detected"}
            </h3>
            {singleResult.confidence !== undefined && (
              <div className="confidence-row">
                <span>Confidence</span>
                <div className="confidence-bar-track">
                  <div
                    className="confidence-bar-fill"
                    style={{
                      width: `${(singleResult.confidence * 100).toFixed(0)}%`,
                      background:
                        singleResult.prediction === 1 ? "#ef4444" : "#22c55e",
                    }}
                  />
                </div>
                <span className="confidence-pct">
                  {(singleResult.confidence * 100).toFixed(1)}%
                </span>
              </div>
            )}
            {singleResult.message && (
              <p className="result-msg">{singleResult.message}</p>
            )}
          </div>
        </div>
      )}

      {/* ── Batch results ── */}
      {batchResults && (
        <div className="card">
          <label className="card-label">
            Batch Results —{" "}
            <span className="preview-count">
              {batchResults.length} patients
            </span>
          </label>
          <div className="table-scroll">
            <table>
              <thead>
                <tr>
                  <th>#</th>
                  <th>Result</th>
                  <th>Confidence</th>
                  <th>Note</th>
                </tr>
              </thead>
              <tbody>
                {batchResults.map((r) => (
                  <tr key={r.row}>
                    <td>{r.row + 1}</td>
                    <td
                      style={{
                        color: r.prediction === 1 ? "#ef4444" : "#22c55e",
                        fontWeight: 600,
                      }}
                    >
                      {r.prediction === 1 ? "At Risk" : "No Risk"}
                    </td>
                    <td>{(r.confidence * 100).toFixed(1)}%</td>
                    <td>{r.message}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* ── Error ── */}
      {status === "error" && (
        <div className="alert alert-error">
          <span>❌</span>
          <div>
            <strong>Request Failed</strong>
            <p>{result?.error || "Unable to reach the diagnosis server."}</p>
          </div>
        </div>
      )}

      <div className="action-row">
        <button
          className="btn btn-primary"
          disabled={
            (mode === "form" && !formComplete) ||
            (mode === "csv" && !file) ||
            status === "loading"
          }
          onClick={handleSubmit}
        >
          {status === "loading" ? (
            <>
              <span className="spinner" /> Analyzing...
            </>
          ) : (
            "Get Diagnosis"
          )}
        </button>
        {status === "done" && (
          <button className="btn btn-ghost" onClick={reset}>
            Clear
          </button>
        )}
      </div>
    </div>
  );
}
