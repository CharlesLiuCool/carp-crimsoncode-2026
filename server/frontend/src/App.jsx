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

  return (
    <div className="app">
      <header className="header">
        <div className="header-inner">
          <div className="header-brand">
            <div>
              <h1 className="header-title">CARP</h1>
              <p className="header-sub">Diabetes Risk Screening</p>
            </div>
          </div>
        </div>
      </header>

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
