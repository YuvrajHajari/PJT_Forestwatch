export default function RiskPanel({ data }) {
  return (
    <>
      <div className="panel-section">
        <div className="section-eyebrow">Deforestation risk</div>
        <div className="big-stat">
          <div className="big-stat-value loss-color">{data.high_risk_area_ha}</div>
          <div className="big-stat-unit">hectares in the top {(100 - data.high_risk_percentile).toFixed(0)}% risk band</div>
          <div className="big-stat-label">
            {data.high_risk_pct_of_forest}% of significant forest area, ranked by the
            CA-ANN's calibrated change-likelihood score
          </div>
        </div>
        <div className="metric-grid">
          <div className="metric-cell">
            <div className="metric-cell-label">Significant forest</div>
            <div className="metric-cell-value">{data.significant_forest_area_ha}</div>
            <div className="metric-cell-unit">ha (≥0.1ha patches)</div>
          </div>
          <div className="metric-cell">
            <div className="metric-cell-label">Mean relative risk</div>
            <div className="metric-cell-value">{data.mean_relative_risk_score}</div>
            <div className="metric-cell-unit">0–1 scale</div>
          </div>
        </div>
      </div>

      <div className="panel-section">
        <div className="hansen-note">{data.note}</div>
      </div>
    </>
  );
}
