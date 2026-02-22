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

const METRIC_LABELS = [
  { key: "accuracy", label: "Accuracy", color: "var(--accent, #2563eb)" },
  { key: "precision", label: "Precision", color: "var(--success, #16a34a)" },
  { key: "recall", label: "Recall", color: "var(--warning, #ca8a04)" },
  { key: "f1", label: "F1", color: "var(--info, #0891b2)" },
];

export default function UploadLogTab() {
  const [entries, setEntries] = useState([]);
  const [metricsHistory, setMetricsHistory] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  const fetchLog = useCallback(async () => {
    setLoading(true);
    setError("");
    try {
      const [listRes, metricsRes] = await Promise.all([
        fetch("/api/weights/list"),
        fetch("/api/weights/metrics-history?limit=3"),
      ]);
      if (!listRes.ok) throw new Error(`Server responded with ${listRes.status}`);
      const listData = await listRes.json();
      setEntries(listData.weights || []);
      if (metricsRes.ok) {
        const metricsData = await metricsRes.json();
        setMetricsHistory(metricsData.metrics || []);
      } else {
        setMetricsHistory([]);
      }
    } catch (err) {
      setError(err.message || "Failed to load upload log.");
      setEntries([]);
      setMetricsHistory([]);
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

      {/* Line graph: metrics over last aggregations (oldest → newest) */}
      {!error && metricsHistory.length > 0 && (() => {
        const points = [...metricsHistory].reverse();
        const n = points.length;
        const chartWidth = 480;
        const chartHeight = 200;
        const pad = { left: 44, right: 24, top: 20, bottom: 36 };
        const plotW = chartWidth - pad.left - pad.right;
        const plotH = chartHeight - pad.top - pad.bottom;
        const x = (i) => (n <= 1 ? pad.left + plotW / 2 : pad.left + (i / Math.max(1, n - 1)) * plotW);
        const y = (v) => pad.top + (1 - Number(v)) * plotH;

        const pathFor = (key) => {
          const coords = points.map((p, i) => `${x(i)},${y(p[key] ?? 0)}`);
          return n > 1 ? `M ${coords.join(" L ")}` : `M ${coords[0]}`;
        };

        return (
          <div className="card" style={{ marginTop: 16 }}>
            <div className="card-label" style={{ display: "flex", alignItems: "center", justifyContent: "space-between", flexWrap: "wrap", gap: 8 }}>
              <span>Model metrics over last {metricsHistory.length} aggregation{metricsHistory.length !== 1 ? "s" : ""}</span>
              <button
                type="button"
                className="btn btn-ghost"
                onClick={fetchLog}
                disabled={loading}
              >
                Refresh
              </button>
            </div>
            <p className="section-sub" style={{ marginTop: 4, marginBottom: 12 }}>
              Accuracy, precision, recall, and F1 after each aggregation. Lines show how metrics change across uploads.
            </p>
            <div style={{ overflowX: "auto" }}>
              <svg
                viewBox={`0 0 ${chartWidth} ${chartHeight}`}
                preserveAspectRatio="xMidYMid meet"
                style={{ width: "100%", maxWidth: chartWidth, height: "auto", minHeight: chartHeight }}
                aria-label="Line chart of model metrics over uploads"
              >
                {/* Y-axis grid & label */}
                {[0, 0.25, 0.5, 0.75, 1].map((v) => (
                  <line
                    key={v}
                    x1={pad.left}
                    y1={y(v)}
                    x2={pad.left + plotW}
                    y2={y(v)}
                    stroke="var(--gray-200)"
                    strokeDasharray="4 4"
                    strokeWidth={1}
                  />
                ))}
                <text x={12} y={pad.top + plotH / 2} textAnchor="middle" fill="var(--gray-500)" fontSize="10" transform={`rotate(-90, 12, ${pad.top + plotH / 2})`}>
                  Score (0–1)
                </text>
                {/* Lines for each metric */}
                {METRIC_LABELS.map(({ key, label, color }) => (
                  <path
                    key={key}
                    d={pathFor(key)}
                    fill="none"
                    stroke={color}
                    strokeWidth={2.5}
                    strokeLinecap="round"
                    strokeLinejoin="round"
                  />
                ))}
                {/* Points on lines */}
                {METRIC_LABELS.map(({ key, color }) =>
                  points.map((p, i) => (
                    <circle
                      key={`${key}-${i}`}
                      cx={x(i)}
                      cy={y(p[key] ?? 0)}
                      r={4}
                      fill={color}
                      stroke="var(--bg, #fff)"
                      strokeWidth={1.5}
                    />
                  ))
                )}
                {/* X-axis labels */}
                {points.map((point, idx) => (
                  <text
                    key={point.id ?? idx}
                    x={x(idx)}
                    y={chartHeight - 10}
                    textAnchor="middle"
                    fill="var(--gray-500)"
                    fontSize="11"
                  >
                    After upload {idx + 1}
                  </text>
                ))}
              </svg>
            </div>
            <div style={{ display: "flex", gap: 16, marginTop: 12, paddingTop: 12, borderTop: "1px solid var(--gray-200)", flexWrap: "wrap" }}>
              {METRIC_LABELS.map(({ key, label, color }) => (
                <span key={key} style={{ display: "flex", alignItems: "center", gap: 6, fontSize: "0.8rem" }}>
                  <span style={{ width: 10, height: 10, borderRadius: 2, backgroundColor: color }} />
                  {label}
                </span>
              ))}
            </div>
          </div>
        );
      })()}
    </div>
  );
}
