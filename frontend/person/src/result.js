export default function ResultCard({ result, onReset }) {
  const { at_risk, probability, risk_level } = result;

  const riskConfig = {
    Low:    { color: "var(--green)", icon: "✓", label: "LOW RISK" },
    Medium: { color: "var(--amber)", icon: "◆", label: "MEDIUM RISK" },
    High:   { color: "var(--red)",   icon: "✕", label: "HIGH RISK" },
  };

  const cfg = riskConfig[risk_level] || riskConfig["Low"];
  const pct = Math.min(probability, 100);

  const message = at_risk
    ? "This assessment suggests an elevated risk for diabetes. Please consult a healthcare professional."
    : "This assessment does not indicate elevated diabetes risk. Maintain a healthy lifestyle and schedule regular checkups.";

  return (
    <div className="result-card">
      <div className="result-hero" style={{ "--risk-color": cfg.color }}>
        <div className="risk-icon-wrap">
          <div className="risk-pulse-ring" />
          <div className="risk-icon">{cfg.icon}</div>
        </div>
        <h2 className="risk-label">{cfg.label}</h2>
        <p className="risk-desc">
          {at_risk
            ? "Elevated diabetes risk detected"
            : "No elevated diabetes risk detected"}
        </p>
      </div>

      <div className="result-body">
        <div className="prob-section">
          <div className="prob-header">
            <span className="prob-title">Risk Probability</span>
            <span className="prob-value" style={{ color: cfg.color }}>
              {pct.toFixed(1)}%
            </span>
          </div>
          <div className="prob-track">
            <div
              className="prob-fill"
              style={{ width: `${pct}%`, background: cfg.color }}
            />
          </div>
          <div className="prob-markers">
            <span>0%</span>
            <span>30%</span>
            <span>60%</span>
            <span>100%</span>
          </div>
          <div className="risk-zones">
            <span className="zone zone-low" />
            <span className="zone zone-medium" />
            <span className="zone zone-high" />
          </div>
        </div>

        <div className="result-message">
          <p>{message}</p>
        </div>

        <div className="disclaimer-box">
          <p className="disclaimer-title">Important Notice</p>
          <p className="disclaimer-text">
            this is for informational purposes only please consult a professional for medical advice.
          </p>
        </div>

        <button className="reset-btn" onClick={onReset}>
          ← Start New Assessment
        </button>
      </div>
    </div>
  );
}