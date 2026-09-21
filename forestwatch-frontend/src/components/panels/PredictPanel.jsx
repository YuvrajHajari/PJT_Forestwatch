import FragmentationGrid from "../FragmentationGrid";
import { predictionNarrative } from "../../narrative";

export default function PredictPanel({ data }) {
  const net = Math.round((data.projected_data.forest_area_ha - data.base_year_data.forest_area_ha) * 100) / 100;
  const netClass = net < 0 ? "loss-color" : "gain-color";
  const v = data.validation;

  return (
    <>
      <div className="panel-section">
        <div className="section-eyebrow">Summary</div>
        <p className="narrative">{predictionNarrative(data)}</p>
      </div>

      <div className="panel-section">
        <div className="validation-banner">
          <div className="vb-label">Model validation</div>
          <div className="vb-value">{v.summary || v.reason || "Validation unavailable."}</div>
        </div>
      </div>

      <div className="panel-section">
        <div className="section-eyebrow">Projected net change</div>
        <div className="big-stat">
          <div className={`big-stat-value ${netClass}`}>{Math.abs(net)}</div>
          <div className="big-stat-unit">hectares projected change</div>
          <div className="big-stat-label">
            {net < 0 ? "projected loss" : "projected gain"} from {data.base_year} to {data.target_year}
          </div>
        </div>
      </div>

      <div className="panel-section">
        <div className="section-eyebrow">Base year — {data.base_year} (measured)</div>
        <div className="metric-grid">
          <div className="metric-cell">
            <div className="metric-cell-label">Forest area</div>
            <div className="metric-cell-value">{data.base_year_data.forest_area_ha}</div>
            <div className="metric-cell-unit">ha</div>
          </div>
          <div className="metric-cell">
            <div className="metric-cell-label">CO₂ stored</div>
            <div className="metric-cell-value">
              {data.base_year_data.co2_equiv_t.toLocaleString("en-IN")}
            </div>
            <div className="metric-cell-unit">t-CO₂e</div>
          </div>
        </div>
        <FragmentationGrid fragmentation={data.base_year_data.fragmentation} />
      </div>

      <div className="panel-section">
        <div className="section-eyebrow">Target year — {data.target_year} (projected)</div>
        <div className="metric-grid">
          <div className="metric-cell">
            <div className="metric-cell-label">Forest area</div>
            <div className="metric-cell-value">{data.projected_data.forest_area_ha}</div>
            <div className="metric-cell-unit">ha</div>
          </div>
          <div className="metric-cell">
            <div className="metric-cell-label">CO₂ stored</div>
            <div className="metric-cell-value">
              {data.projected_data.co2_equiv_t.toLocaleString("en-IN")}
            </div>
            <div className="metric-cell-unit">t-CO₂e</div>
          </div>
        </div>
        <FragmentationGrid fragmentation={data.projected_data.fragmentation} />
      </div>

      <div className="panel-section">
        <div className="section-eyebrow">Learned annual rates</div>
        <div className="change-row">
          <span className="change-key">P(non-forest → forest)</span>
          <span className="change-val pos">{(data.projection_meta.p_gain_annual * 100).toFixed(3)}%/yr</span>
        </div>
        <div className="change-row">
          <span className="change-key">P(forest → non-forest)</span>
          <span className="change-val neg">{(data.projection_meta.p_loss_annual * 100).toFixed(3)}%/yr</span>
        </div>
      </div>
    </>
  );
}
