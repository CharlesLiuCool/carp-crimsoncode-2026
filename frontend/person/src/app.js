import { useState } from "react";
import DiabetesForm from "./form";
import ResultCard from "./result";
import "./app.css";

export default function App() {
  const [result, setResult]   = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError]     = useState(null);

  const handleSubmit = async (formData) => {
    setLoading(true);
    setError(null);
    setResult(null);
    



    try {
      const res = await fetch("http://localhost:5000/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(formData),
      });

      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.error || "Server error");
      }

      const data = await res.json();
      setResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setResult(null);
    setError(null);
  };

  return (
    <div className="app">
      <header className="app-header">
        <div className="header-inner">
          <div className="logo-mark">
            <span className="logo-pulse" />
            <span className="logo-text">carp</span>
          </div>
          <p className="header-tagline">AI-Powered Diabetes Risk Screening</p>
        </div>
      </header>

      <main className="app-main">
        {!result ? (
          <DiabetesForm onSubmit={handleSubmit} loading={loading} error={error} />
        ) : (
          <ResultCard result={result} onReset={handleReset} />
        )}
      </main>

      <footer className="app-footer">
        <p>For informational purposes only. Not a substitute for medical advice.</p>
      </footer>
    </div>
  );
}