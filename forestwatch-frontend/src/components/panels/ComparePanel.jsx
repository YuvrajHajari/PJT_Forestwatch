export default function ComparePanel({ data }) {
  const net = data.change.net_ha;
  const netClass = net < 0 ? "loss-color" : "gain-color";

  return (
    <>
      <div className="panel-section">
        <div className="section-eyebrow">Net change</div>
        <div className="big-stat">
          <div className={`big-stat-value ${netClass}`}>{Math.abs(net)}</div>
          <div className="big-stat-unit">hectares net change</div>
          <div className="big-stat-label">
            {net < 0 ? "lost" : "gained"} between {data.year_a} and {data.year_b}
          </div>
        </div>
      </div>

      <div className="panel-section">
        <div className="section-eyebrow">Breakdown</div>
        <div className="change-row">
          <span className="change-key">Forest lost</span>
          <span className="change-val neg">-{data.change.lost_ha} ha</span>
        </div>
        <div className="change-row">
          <span className="change-key">Forest gained</span>
          <span className="change-val pos">+{data.change.gained_ha} ha</span>
        </div>
        <div className="change-row">
          <span className="change-key">Carbon released</span>
          <span className="change-val neg">-{data.change.lost_carbon_t} t-C</span>
        </div>
        <div className="change-row">
          <span className="change-key">CO₂ equivalent</span>
          <span className="change-val neg">-{data.change.lost_co2_t} t-CO₂e</span>
        </div>
      </div>

      <div className="panel-section">
        <div className="section-eyebrow">Year A — {data.year_a}</div>
        <div className="metric-grid">
          <div className="metric-cell">
            <div className="metric-cell-label">Forest area</div>
            <div className="metric-cell-value">{data.year_a_data.forest_area_ha}</div>
            <div className="metric-cell-unit">ha</div>
          </div>
          <div className="metric-cell">
            <div className="metric-cell-label">CO₂ stored</div>
            <div className="metric-cell-value">{data.year_a_data.co2_equiv_t.toLocaleString("en-IN")}</div>
            <div className="metric-cell-unit">t-CO₂e</div>
          </div>
        </div>
      </div>

      <div className="panel-section">
        <div className="section-eyebrow">Year B — {data.year_b}</div>
        <div className="metric-grid">
          <div className="metric-cell">
            <div className="metric-cell-label">Forest area</div>
            <div className="metric-cell-value">{data.year_b_data.forest_area_ha}</div>
            <div className="metric-cell-unit">ha</div>
          </div>
          <div className="metric-cell">
            <div className="metric-cell-label">CO₂ stored</div>
            <div className="metric-cell-value">{data.year_b_data.co2_equiv_t.toLocaleString("en-IN")}</div>
            <div className="metric-cell-unit">t-CO₂e</div>
          </div>
        </div>
      </div>
    </>
  );
}
