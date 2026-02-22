import { useState } from "react";
import "./App.css";
import DiagnosisTab from "./components/DiagnosisTab";
import WeightsTab from "./components/WeightsTab";

// Add new tab labels here as the server frontend grows.
// Each entry needs a corresponding tab panel rendered in <main> below.
const TABS = [
  { label: "Patient Diagnosis", component: <DiagnosisTab /> },
  { label: "Upload Weights", component: <WeightsTab /> },
  // { label: "Analytics",      component: <AnalyticsTab /> },
];

export default function App() {
  const [activeTab, setActiveTab] = useState(0);
  const [helpOpen, setHelpOpen] = useState(false);

  return (
    <div className="app">
      <header className="header">
        <div className="header-inner">
          <div className="header-brand">
            <div className="header-logo-wrap">
              <img src="/carp-logo.png" alt="CARP Tech" className="header-logo" />
            </div>
            <div>
              <h1 className="header-title">CARP</h1>
              <p className="header-sub">Diabetes Risk Screening</p>
            </div>
          </div>
          <button
            type="button"
            className="header-help-btn"
            onClick={() => setHelpOpen(true)}
            title="About & how to use"
            aria-label="About and how to use this site"
          >
            ?
          </button>
        </div>
      </header>

      {helpOpen && (
        <div className="help-overlay" onClick={() => setHelpOpen(false)} aria-hidden="true">
          <div className="help-modal" onClick={(e) => e.stopPropagation()}>
            <div className="help-modal-header">
              <h2>About the CARP Server</h2>
              <button
                type="button"
                className="help-close"
                onClick={() => setHelpOpen(false)}
                aria-label="Close"
              >
                ×
              </button>
            </div>
            <div className="help-modal-body">
              <p>
                This is the <strong>central CARP server</strong>. It aggregates differentially private (DP) model weights from hospital clients and provides diabetes risk predictions using the combined model.
              </p>
              <h3>How to use</h3>
              <p>
                <strong>Patient Diagnosis:</strong> Enter a patient’s Age, BMI, and Glucose to get a diabetes risk prediction from the <strong>aggregated central model</strong>. The model is updated whenever new weights are uploaded.
              </p>
              <p>
                <strong>Upload Weights:</strong> Hospital clients train locally with DP and export their weights. Upload those <code>.pt</code> files here. The server averages all uploaded weights and updates the central model so diagnosis uses the latest combined model.
              </p>
            </div>
          </div>
        </div>
      )}

      <main className="main">
        <div className="container">
          {/* Tab bar — only rendered when there is more than one tab */}
          {TABS.length > 1 && (
            <div className="tabs">
              {TABS.map((tab, i) => (
                <button
                  key={tab.label}
                  className={`tab-btn ${activeTab === i ? "tab-active" : ""}`}
                  onClick={() => setActiveTab(i)}
                >
                  {tab.label}
                </button>
              ))}
            </div>
          )}

          {TABS[activeTab].component}
        </div>
      </main>
    </div>
  );
}
