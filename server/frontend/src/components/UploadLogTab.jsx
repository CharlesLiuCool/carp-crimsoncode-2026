import "../App.css";
import { useState, useEffect, useCallback } from "react";

function formatDate(iso) {
  if (!iso) return "—";
  try {
    const d = new Date(iso);
    return d.toLocaleString(undefined, {
      dateStyle: "medium",
      timeStyle: "short",
    });
  } catch {
    return iso;
  }
}

export default function UploadLogTab() {
  const [entries, setEntries] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  const fetchLog = useCallback(async () => {
    setLoading(true);
    setError("");
    try {
      const res = await fetch("/api/weights/list");
      if (!res.ok) throw new Error(`Server responded with ${res.status}`);
      const data = await res.json();
      setEntries(data.weights || []);
    } catch (err) {
      setError(err.message || "Failed to load upload log.");
      setEntries([]);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchLog();
  }, [fetchLog]);

  return (
    <div className="tab-content">
      <div className="section-header">
        <h2>Hospital Upload Log</h2>
        <p className="section-sub">
          Weights files uploaded by hospital clients. This list is used for the
          aggregated central model.
        </p>
      </div>

      <div className="card">
        <div className="card-label" style={{ display: "flex", alignItems: "center", justifyContent: "space-between", flexWrap: "wrap", gap: 8 }}>
          <span>Uploaded weights</span>
          <button
            type="button"
            className="btn btn-ghost"
            onClick={fetchLog}
            disabled={loading}
          >
            {loading ? (
              <>
                <span className="spinner" style={{ width: 14, height: 14 }} /> Loading...
              </>
            ) : (
              "Refresh"
            )}
          </button>
        </div>

        {error && (
          <div className="alert alert-error" style={{ marginTop: 8 }}>
            <p>{error}</p>
          </div>
        )}

        {!error && !loading && entries.length === 0 && (
          <p className="table-more" style={{ whiteSpace: "normal", textAlign: "center", padding: 24 }}>
            No hospital weights uploaded yet. Use the Upload Weights tab to add
            contributions.
          </p>
        )}

        {!error && entries.length > 0 && (
          <div className="table-scroll" style={{ marginTop: 8 }}>
            <table>
              <thead>
                <tr>
                  <th>#</th>
                  <th>Filename</th>
                  <th>Uploaded at</th>
                  <th>Size</th>
                </tr>
              </thead>
              <tbody>
                {entries.map((row, i) => (
                  <tr key={row.id}>
                    <td>{entries.length - i}</td>
                    <td>
                      <code className="file-name">{row.filename}</code>
                    </td>
                    <td>{formatDate(row.uploaded_at)}</td>
                    <td>{row.size_kb != null ? `${Number(row.size_kb).toFixed(1)} KB` : "—"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}
