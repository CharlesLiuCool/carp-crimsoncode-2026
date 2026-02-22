import { useState } from "react";
import "./App.css";
import TrainTab from "./components/TrainTab";
import DiagnosisTab from "./components/DiagnosisTab";

const TABS = ["Train Model", "Patient Diagnosis"];

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
              <h2>About CARP</h2>
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
                <strong>CARP</strong> is a diabetes risk screening tool. It lets you train a local model on patient data (with optional differential privacy) and run risk predictions for individual patients or batches.
              </p>
              <h3>How to use</h3>
              <p>
                <strong>Train Model:</strong> Upload a CSV with columns <code>Age</code>, <code>BMI</code>, <code>Glucose</code>, and <code>Outcome</code>. Use the privacy slider to turn on differential privacy (DP) for stronger privacy guarantees. After training, you can export DP weights to send to a central server.
              </p>
              <p>
                <strong>Patient Diagnosis:</strong> Enter a patient’s Age, BMI, and Glucose, or upload a CSV with those columns, to get a diabetes risk prediction. Run training at least once before using diagnosis.
              </p>
            </div>
          </div>
        </div>
      )}

      <main className="main">
        <div className="container">
          <div className="tabs">
            {TABS.map((tab, i) => (
              <button
                key={tab}
                className={`tab-btn ${activeTab === i ? "tab-active" : ""}`}
                onClick={() => setActiveTab(i)}
              >
                {tab}
              </button>
            ))}
          </div>

          {activeTab === 0 && <TrainTab />}
          {activeTab === 1 && <DiagnosisTab />}
        </div>
      </main>
    </div>
  );
}
