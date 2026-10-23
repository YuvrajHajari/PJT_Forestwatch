import { useEffect, useState } from "react";
import { api, bust } from "../../api";
import ProcessLog from "../ProcessLog";
import ClayFlowDiagram, { BtsSummaryDiagram } from "../diagrams/ClayFlowDiagram";
import PipelineStrip from "../PipelineStrip";
import { stepsFor } from "../../processSteps";

const TARGET_YEARS = ["2028", "2031", "2034"];

export default function PredictSection({ city, cityLabel }) {
  const [tab, setTab] = useState("bts");
  const [targetYear, setTargetYear] = useState(TARGET_YEARS[0]);
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (tab !== "working") return;
    setLoading(true);
    setData(null);
    api
      .predict(city, targetYear)
      .then(setData)
      .finally(() => setLoading(false));
  }, [city, targetYear, tab]);

  const net = data
    ? Math.round((data.projected_data.forest_area_ha - data.base_year_data.forest_area_ha) * 100) / 100
    : 0;

  return (
    <section className="feature-section">
      <div className="section-eyebrow">Forward prediction</div>
      <h2 className="section-title">That's enough computer vision. Here's the AI.</h2>

      <div className="predict-tabs">
        <button className={`predict-tab${tab === "bts" ? " active" : ""}`} onClick={() => setTab("bts")}>
          Behind the scenes
        </button>
        <button className={`predict-tab${tab === "working" ? " active" : ""}`} onClick={() => setTab("working")}>
          See it work
        </button>
      </div>

      {tab === "bts" && (
        <>
          <p className="section-sub">
            Two very different things feed this prediction. One learns, one doesn't. Here's how
            they connect, step by step.
          </p>
          <ClayFlowDiagram />
          <BtsSummaryDiagram />
        </>
      )}

      {tab === "working" && (
        <>
          <PipelineStrip mode="predict" />
          <div className="year-strip">
            {TARGET_YEARS.map((y) => (
              <button key={y} className={`year-chip${targetYear === y ? " active" : ""}`} onClick={() => setTargetYear(y)}>
                {y}
              </button>
            ))}
          </div>

          {loading && (
            <div className="stage-loader embedded">
              <ProcessLog steps={stepsFor("predict", { city: cityLabel, targetYear })} />
            </div>
          )}

          {data && !loading && (
            <div className="two-col diff-row">
              <div className="mask-viewer">
                <img className="mask-viewer-img" src={bust(data.predict_url)} alt="Projection" />
                <div className="img-label left">
                  {cityLabel.toUpperCase()} &middot; PROJECTED {data.target_year}
                </div>
                <div className="diff-legend static-legend">
                  <span><span className="legend-dot" style={{ background: "#1a6b3a" }} />Stable forest</span>
                  <span><span className="legend-dot" style={{ background: "#e8b400" }} />Projected gain</span>
                  <span><span className="legend-dot" style={{ background: "#c0392b" }} />Projected loss</span>
                </div>
              </div>
              <div className="side-stats">
                <div className="validation-banner">
                  <div className="vb-label">Model validation</div>
                  <div className="vb-value">{data.validation.summary || data.validation.reason}</div>
                </div>
                <div className="big-stat">
                  <div className={`big-stat-value ${net < 0 ? "loss-color" : "gain-color"}`}>{Math.abs(net)}</div>
                  <div className="big-stat-unit">ha projected {net < 0 ? "loss" : "gain"}</div>
                  <div className="big-stat-label">{data.base_year} &rarr; {data.target_year}</div>
                </div>
                <div className="change-row">
                  <span className="change-key">P(non-forest &rarr; forest)</span>
                  <span className="change-val pos">{(data.projection_meta.p_gain_annual * 100).toFixed(3)}%/yr</span>
                </div>
                <div className="change-row">
                  <span className="change-key">P(forest &rarr; non-forest)</span>
                  <span className="change-val neg">{(data.projection_meta.p_loss_annual * 100).toFixed(3)}%/yr</span>
                </div>
                <div className="change-row">
                  <span className="change-key">Method</span>
                  <span className="change-val">{data.projection_meta.method}</span>
                </div>
              </div>
            </div>
          )}
        </>
      )}
    </section>
  );
}
