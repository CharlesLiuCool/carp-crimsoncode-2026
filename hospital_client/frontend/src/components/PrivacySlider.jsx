import "../App.css";
import { useState, useEffect } from "react";

const MIN = 0.1;
const MAX = 10.0;
const HIPAA = 1.0;

function pct(epsilon) {
  return ((epsilon - MIN) / (MAX - MIN)) * 100;
}

function clamp(val) {
  return Math.min(MAX, Math.max(MIN, val));
}

function getBadge(epsilon) {
  if (Math.abs(epsilon - HIPAA) < 0.05)
    return { cls: "badge-hipaa", label: "★ HIPAA Recommended" };
  if (epsilon < 1.0) return { cls: "badge-privacy", label: "Maximum Privacy" };
  if (epsilon < 3.0) return { cls: "badge-privacy", label: "Strong Privacy" };
  if (epsilon < 7.0) return { cls: "badge-balanced", label: "Balanced" };
  return { cls: "badge-accuracy", label: "Maximum Accuracy" };
}

function getHint(epsilon) {
  if (Math.abs(epsilon - HIPAA) < 0.05)
    return "ε = 1.0 — the recommended threshold for HIPAA-sensitive medical data. Strong privacy guarantee with acceptable model performance.";
  if (epsilon < 1.0)
    return "Very strong differential privacy. Maximum patient protection — model accuracy will be significantly reduced.";
  if (epsilon < 3.0)
    return "Strong differential privacy. Patient data is well protected but model accuracy may be reduced.";
  if (epsilon < 7.0)
    return "Balanced trade-off between privacy guarantees and model performance.";
  return "High accuracy training. Less noise added — ensure this complies with your institution's HIPAA policy.";
}

export default function PrivacySlider({ value, onChange }) {
  const [draft, setDraft] = useState(value.toFixed(2));

  // Keep draft in sync when value changes externally (e.g. slider drag)
  useEffect(() => {
    setDraft(value.toFixed(2));
  }, [value]);

  function handleSlider(e) {
    const v = clamp(parseFloat(e.target.value));
    onChange(v);
  }

  function handleInputChange(e) {
    setDraft(e.target.value);
  }

  function handleInputCommit() {
    const parsed = parseFloat(draft);
    if (!isNaN(parsed)) {
      const clamped = clamp(parsed);
      onChange(clamped);
      setDraft(clamped.toFixed(2));
    } else {
      setDraft(value.toFixed(2));
    }
  }

  function handleInputKeyDown(e) {
    if (e.key === "Enter") e.target.blur();
  }

  const fill = pct(value);
  const { cls, label } = getBadge(value);
  const hint = getHint(value);
  const hipaaLeft = `${pct(HIPAA).toFixed(2)}%`;

  return (
    <div className="slider-container">
      <div className="slider-header">
        <span className="slider-label">Privacy / Accuracy Trade-off</span>
        <div className="slider-epsilon-input-row">
          <span className="slider-epsilon-prefix">ε =</span>
          <input
            className="slider-epsilon-input"
            type="number"
            min={MIN}
            max={MAX}
            step="0.01"
            value={draft}
            onChange={handleInputChange}
            onBlur={handleInputCommit}
            onKeyDown={handleInputKeyDown}
          />
        </div>
      </div>

      <div className="slider-track-wrapper">
        <span className="slider-end-label privacy">Privacy</span>
        <div className="slider-track-inner">
          <input
            type="range"
            min={MIN}
            max={MAX}
            step="0.01"
            value={value}
            onChange={handleSlider}
            className="slider"
            style={{
              background: `linear-gradient(to right, #3b82f6 0%, #3b82f6 ${fill}%, #e2e8f0 ${fill}%, #e2e8f0 100%)`,
            }}
          />
          <div className="slider-minmax-row">
            <span className="slider-minmax">{MIN.toFixed(1)}</span>
            <span className="slider-minmax">{MAX.toFixed(1)}</span>
          </div>
        </div>
        <span className="slider-end-label accuracy">Accuracy</span>
      </div>

      {/* HIPAA marker pinned at ε = 1.0 on the track */}
      <div className="slider-hipaa-marker-row">
        <span className="slider-hipaa-marker" style={{ left: hipaaLeft }}>
          ★ ε = 1.0
        </span>
      </div>

      <div className="slider-badge-row">
        <span className={`slider-badge ${cls}`}>{label}</span>
        <p className="slider-hint">{hint}</p>
      </div>
    </div>
  );
}
