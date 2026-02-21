import "../App.css";
import { useState, useRef } from "react";
import { parseCSV, validateColumns, DIAGNOSIS_COLUMNS } from "../utils/csv";
import CSVPreview from "./CSVPreview";

export default function DiagnosisTab() {
  const [mode, setMode] = useState("form"); // 'form' | 'csv'
  const [formData, setFormData] = useState({
    Pregnancies: "",
    Glucose: "",
    BloodPressure: "",
    Insulin: "",
    BMI: "",
    DiabetesPedigreeFunction: "",
    Age: "",
  });
  const [file, setFile] = useState(null);
  const [parsed, setParsed] = useState(null);
  const [csvError, setCsvError] = useState("");
  const [status, setStatus] = useState(null);
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
          body: JSON.stringify(formData),
        });
      } else {
        const fd = new FormData();
        fd.append("file", file);
        res = await fetch("/api/diagnose", { method: "POST", body: fd });
      }
      if (!res.ok) throw new Error(`Server responded with ${res.status}`);
      const data = await res.json();
      setStatus("done");
      setResult(data);
    } catch (err) {
      setStatus("error");
      setResult({ error: err.message });
    }
  }

  const fieldLabels = {
    Pregnancies: {
      label: "Pregnancies",
      unit: "",
      type: "number",
      min: 0,
      max: 20,
    },
    Glucose: {
      label: "Glucose",
      unit: "mg/dL",
      type: "number",
      min: 0,
      max: 300,
    },
    BloodPressure: {
      label: "Blood Pressure",
      unit: "mmHg",
      type: "number",
      min: 0,
      max: 200,
    },
    Insulin: {
      label: "Insulin",
      unit: "μU/mL",
      type: "number",
      min: 0,
      max: 900,
    },
    BMI: {
      label: "BMI",
      unit: "kg/m²",
      type: "number",
      min: 0,
      max: 70,
      step: 0.1,
    },
    DiabetesPedigreeFunction: {
      label: "Diabetes Pedigree Function",
      unit: "",
      type: "number",
      min: 0,
      max: 3,
      step: 0.001,
    },
    Age: { label: "Age", unit: "years", type: "number", min: 1, max: 120 },
  };

  return (
    <div className="tab-content">
      <div className="section-header">
        <h2>Patient Diagnosis</h2>
        <p className="section-sub">
          Submit patient data to the central model to receive a diabetes risk
          assessment.
        </p>
      </div>

      <div className="mode-toggle">
        <button
          className={`mode-btn ${mode === "form" ? "active" : ""}`}
          onClick={() => setMode("form")}
        >
          Manual Entry
        </button>
        <button
          className={`mode-btn ${mode === "csv" ? "active" : ""}`}
          onClick={() => setMode("csv")}
        >
          Upload CSV
        </button>
      </div>

      {mode === "form" && (
        <div className="card">
          <label className="card-label">Patient Information</label>
          <div className="form-grid">
            {DIAGNOSIS_COLUMNS.map((field) => {
              const meta = fieldLabels[field];
              return (
                <div className="form-field" key={field}>
                  <label>{meta.label}</label>
                  <div className="input-wrapper">
                    <input
                      type={meta.type}
                      min={meta.min}
                      max={meta.max}
                      step={meta.step || 1}
                      placeholder="0"
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
                    setFile(null);
                    setParsed(null);
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

      {status === "done" && result && !result.error && (
        <div
          className={`result-card ${result.prediction === 1 ? "result-positive" : "result-negative"}`}
        >
          <div className="result-body">
            <h3>
              {result.prediction === 1
                ? "Diabetes Risk Detected"
                : "No Diabetes Detected"}
            </h3>
            {result.confidence !== undefined && (
              <div className="confidence-row">
                <span>Confidence</span>
                <div className="confidence-bar-track">
                  <div
                    className="confidence-bar-fill"
                    style={{
                      width: `${(result.confidence * 100).toFixed(0)}%`,
                      background:
                        result.prediction === 1 ? "#ef4444" : "#22c55e",
                    }}
                  />
                </div>
                <span className="confidence-pct">
                  {(result.confidence * 100).toFixed(1)}%
                </span>
              </div>
            )}
            {result.message && <p className="result-msg">{result.message}</p>}
          </div>
        </div>
      )}

      {status === "error" && (
        <div className="alert alert-error">
          <span>❌</span>
          <div>
            <strong>Request Failed</strong>
            <p>{result?.error || "Unable to reach the central server."}</p>
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
      </div>
    </div>
  );
}
