import { useState } from "react";
import "./App.css";
import TrainTab from "./components/TrainTab";
import DiagnosisTab from "./components/DiagnosisTab";

const TABS = ["Train Model", "Patient Diagnosis"];

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
