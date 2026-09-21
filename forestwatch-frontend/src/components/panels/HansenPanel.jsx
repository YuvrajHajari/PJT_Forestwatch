export default function HansenPanel({ data }) {
  const vm = data.validation_metrics;
  const ba = data.buffered_agreement;
  const maxArea = Math.max(data.our_forest_area_ha, data.hansen_forest_area_ha, 1);

  return (
    <>
      <div className="panel-section">
        <div className="section-eyebrow">Forest area comparison</div>
        <div className="hansen-compare-bars">
          <div className="bar-row">
            <span className="bar-year">Ours</span>
            <div className="bar-track">
              <div
                className="bar-fill"
                style={{ width: `${(data.our_forest_area_ha / maxArea) * 100}%` }}
              />
            </div>
            <span className="bar-val">{data.our_forest_area_ha} ha</span>
          </div>
          <div className="bar-row">
            <span className="bar-year">Hansen</span>
            <div className="bar-track">
              <div
                className="bar-fill"
                style={{
                  width: `${(data.hansen_forest_area_ha / maxArea) * 100}%`,
                  background: "var(--accent)",
                }}
              />
            </div>
            <span className="bar-val">{data.hansen_forest_area_ha} ha</span>
          </div>
        </div>
        <p style={{ fontSize: 11, color: "var(--ink3)", marginTop: 10 }}>
          Canopy threshold: {data.canopy_threshold_pct}%
        </p>
      </div>

      <div className="panel-section">
        <div className="section-eyebrow">Strict pixel-exact accuracy</div>
        <div className="hansen-metric-grid">
          <div className="hansen-metric-cell">
            <div className="hansen-metric-cell-label">Precision</div>
            <div className="hansen-metric-cell-value">{vm.precision}</div>
          </div>
          <div className="hansen-metric-cell">
            <div className="hansen-metric-cell-label">Recall</div>
            <div className="hansen-metric-cell-value">{vm.recall}</div>
          </div>
          <div className="hansen-metric-cell">
            <div className="hansen-metric-cell-label">F1</div>
            <div className="hansen-metric-cell-value">{vm.f1_score}</div>
          </div>
          <div className="hansen-metric-cell">
            <div className="hansen-metric-cell-label">Kappa</div>
            <div className="hansen-metric-cell-value">{vm.kappa}</div>
          </div>
        </div>
      </div>

      <div className="panel-section">
        <div className="section-eyebrow">Buffered agreement ({ba.buffer_m}m tolerance)</div>
        <div className="hansen-metric-grid">
          <div className="hansen-metric-cell">
            <div className="hansen-metric-cell-label">Precision</div>
            <div className="hansen-metric-cell-value">{ba.buffered_precision}</div>
          </div>
          <div className="hansen-metric-cell">
            <div className="hansen-metric-cell-label">Recall</div>
            <div className="hansen-metric-cell-value">{ba.buffered_recall}</div>
          </div>
          <div className="hansen-metric-cell" style={{ gridColumn: "span 2" }}>
            <div className="hansen-metric-cell-label">F1</div>
            <div className="hansen-metric-cell-value">{ba.buffered_f1}</div>
          </div>
        </div>
      </div>

      <div className="panel-section">
        <div className="section-eyebrow">Methodology</div>
        <div className="hansen-note">{data.methodology_note}</div>
        <div className="hansen-note" style={{ marginTop: 10 }}>
          {data.caveat}
        </div>
      </div>
    </>
  );
}
