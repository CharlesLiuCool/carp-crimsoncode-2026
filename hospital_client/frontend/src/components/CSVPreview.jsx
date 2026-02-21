import "../App.css";

export default function CSVPreview({ headers, rows }) {
  const preview = rows.slice(0, 5);
  return (
    <div className="csv-preview">
      <p className="preview-label">
        Preview <span className="preview-count">({rows.length} rows)</span>
      </p>
      <div className="table-scroll">
        <table>
          <thead>
            <tr>
              {headers.map((h) => (
                <th key={h}>{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {preview.map((row, i) => (
              <tr key={i}>
                {row.map((val, j) => (
                  <td key={j}>{val}</td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
        {rows.length > 5 && (
          <p className="table-more">+ {rows.length - 5} more rows</p>
        )}
      </div>
    </div>
  );
}
