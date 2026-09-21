export default function ConservationPanel({ data }) {
  const patches = data.priority_patches || [];
  const maxScore = Math.max(...patches.map((p) => p.priority_score), 0.0001);

  return (
    <div className="panel-section">
      <div className="section-eyebrow">Priority patches</div>
      <p style={{ fontSize: 12, color: "var(--ink3)", marginBottom: 14, lineHeight: 1.6 }}>
        Ranked by patch size combined with the CA-ANN's mean risk score — the
        numbers on the map correspond to the ranks below.
      </p>
      <div className="priority-list">
        {patches.map((p, i) => (
          <div className="priority-row" key={p.patch_id}>
            <div className="priority-rank">{i + 1}</div>
            <div className="priority-body">
              <div className="priority-top-row">
                <span className="priority-area">{p.area_ha} ha</span>
                <span className="priority-score">score {p.priority_score}</span>
              </div>
              <div className="priority-risk-bar">
                <div
                  className="priority-risk-fill"
                  style={{ width: `${(p.priority_score / maxScore) * 100}%` }}
                />
              </div>
              <div style={{ fontFamily: "var(--mono)", fontSize: 10, color: "var(--ink3)", marginTop: 4 }}>
                mean risk {p.mean_risk}
              </div>
            </div>
          </div>
        ))}
        {patches.length === 0 && (
          <p style={{ fontSize: 12, color: "var(--ink3)" }}>
            No patches above the minimum size threshold were found.
          </p>
        )}
      </div>
    </div>
  );
}
