import "../App.css";
import { useState } from "react";
import { DIAGNOSIS_COLUMNS } from "../utils/csv";

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
  const [formData, setFormData] = useState({ Age: "", BMI: "", Glucose: "" });
  const [status, setStatus] = useState(null); // null | 'loading' | 'done' | 'error'
  const [result, setResult] = useState(null);
  const [exportError, setExportError] = useState(null);

  const formComplete = DIAGNOSIS_COLUMNS.every((k) => formData[k] !== "");

  function handleFormChange(field, value) {
    setFormData((prev) => ({ ...prev, [field]: value }));
  }

  function reset() {
    setFormData({ Age: "", BMI: "", Glucose: "" });
    setStatus(null);
    setResult(null);
    setExportError(null);
  }

  async function handleExportModel() {
    setExportError(null);
    try {
      const res = await fetch("/api/model/export");
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || `Export failed (${res.status})`);
      }
      const blob = await res.blob();
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = "carp_central_model.pkl";
      a.click();
      URL.revokeObjectURL(url);
    } catch (err) {
      setExportError(err.message || "Failed to export model.");
    }
  }

  async function handleSubmit() {
    setStatus("loading");
    setResult(null);
    try {
      const res = await fetch("/api/diagnose", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          Age: Number(formData.Age),
          BMI: Number(formData.BMI),
          Glucose: Number(formData.Glucose),
        }),
      });
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

  return (
    <div className="tab-content">
      <div className="section-header">
        <h2>Patient Diagnosis</h2>
        <p className="section-sub">
          Enter patient vitals to receive a diabetes risk assessment from the
          trained model.
        </p>
      </div>

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

      {/* ── Result ── */}
      {status === "done" && result && !result.error && (
        <div
          className={`result-card ${result.prediction === 1 ? "result-positive" : "result-negative"}`}
        >
          <div className="result-body">
            <h3>
              {result.prediction === 1
                ? "Diabetes Risk Detected"
                : "No Diabetes Risk Detected"}
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

      {/* ── SHAP / Feature contributions ── */}
      {status === "done" && result?.feature_contributions && (
        <div className="card shap-card">
          <label className="card-label">
            How each factor contributed to your risk score
          </label>
          <p className="section-sub" style={{ marginTop: 0 }}>
            Positive = increased risk, negative = decreased risk. The AI analysis
            below references these contributions.
          </p>
          <div className="shap-list">
            {["Age", "BMI", "Glucose"].map((name) => {
              const value = result.feature_contributions[name];
              if (value == null) return null;
              const maxAbs = Math.max(
                ...Object.values(result.feature_contributions).map(Math.abs),
                0.001
              );
              const pct = (value / maxAbs) * 50;
              const isPos = value >= 0;
              return (
                <div key={name} className="shap-row">
                  <span className="shap-label">{name}</span>
                  <div className="shap-bar-track">
                    <div
                      className={`shap-bar ${isPos ? "shap-bar-pos" : "shap-bar-neg"}`}
                      style={{
                        width: `${Math.min(50, Math.abs(pct))}%`,
                        ...(isPos ? { left: "50%" } : { right: "50%" }),
                      }}
                    />
                  </div>
                  <span
                    className={`shap-value ${isPos ? "shap-value-pos" : "shap-value-neg"}`}
                  >
                    {value > 0 ? "+" : ""}
                    {Number(value).toFixed(3)}
                  </span>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* ── LLM Analysis ── */}
      {status === "done" && result?.analysis && (
        <div className="card">
          <label className="card-label">
            AI Clinical Analysis
            {result.analysis_provider && (
              <span className="preview-count">
                {" "}
                — {result.analysis_provider}
              </span>
            )}
          </label>
          <p
            style={{
              fontSize: "0.875rem",
              color: "var(--gray-700)",
              lineHeight: 1.7,
              whiteSpace: "pre-wrap",
            }}
          >
            {result.analysis}
          </p>
          <p
            style={{
              fontSize: "0.72rem",
              color: "var(--gray-400)",
              marginTop: 8,
            }}
          >
            {result.analysis_provider
              ? `Generated by ${result.analysis_provider}.`
              : "AI-generated analysis."}{" "}
            For informational purposes only — not a substitute for clinical
            judgement.
          </p>
        </div>
      )}

      {exportError && (
        <div className="alert alert-error">
          <div>
            <strong>Export failed</strong>
            <p>{exportError}</p>
          </div>
        </div>
      )}

      {/* ── Error ── */}
      {status === "error" && (
        <div className="alert alert-error">
          <div>
            <strong>Request Failed</strong>
            <p>{result?.error || "Unable to reach the diagnosis server."}</p>
          </div>
        </div>
      )}

      <div className="action-row">
        <button
          className="btn btn-primary"
          disabled={!formComplete || status === "loading"}
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
        <button
          type="button"
          className="btn btn-ghost"
          onClick={handleExportModel}
          title="Download aggregated central model as .pkl"
        >
          Export model
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
