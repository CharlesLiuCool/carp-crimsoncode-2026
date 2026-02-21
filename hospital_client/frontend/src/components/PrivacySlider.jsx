import "../App.css";

export default function PrivacySlider({ value, onChange }) {
  const labels = ["Maximum Privacy", "Balanced", "Maximum Accuracy"];
  const labelIndex = value <= 3 ? 0 : value <= 7 ? 1 : 2;
  const epsilon = ((value / 10) * 9.9 + 0.1).toFixed(2);

  return (
    <div className="slider-container">
      <div className="slider-header">
        <span className="slider-label">Privacy / Accuracy Trade-off</span>
        <span className="slider-epsilon">ε = {epsilon}</span>
      </div>
      <div className="slider-track-wrapper">
        <span className="slider-end-label privacy">Privacy</span>
        <input
          type="range"
          min="0"
          max="10"
          step="1"
          value={value}
          onChange={(e) => onChange(Number(e.target.value))}
          className="slider"
          style={{
            background: `linear-gradient(to right, #3b82f6 0%, #3b82f6 ${value * 10}%, #e2e8f0 ${value * 10}%, #e2e8f0 100%)`,
          }}
        />
        <span className="slider-end-label accuracy">Accuracy</span>
      </div>
      <div className="slider-ticks">
        {[...Array(11)].map((_, i) => (
          <span key={i} className={`tick ${i === value ? "active" : ""}`} />
        ))}
      </div>
      <div className="slider-badge-row">
        <span
          className={`slider-badge ${labelIndex === 0 ? "badge-privacy" : labelIndex === 2 ? "badge-accuracy" : "badge-balanced"}`}
        >
          {labels[labelIndex]}
        </span>
        <p className="slider-hint">
          {value <= 3
            ? "Strong differential privacy. Patient data is highly protected but model accuracy may be reduced."
            : value <= 7
              ? "Balanced trade-off between privacy guarantees and model performance."
              : "High accuracy training. Less noise added."}
        </p>
      </div>
    </div>
  );
}
