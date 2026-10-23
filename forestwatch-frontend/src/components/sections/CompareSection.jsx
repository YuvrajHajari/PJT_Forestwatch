import { useEffect, useState } from "react";
import { api, bust } from "../../api";
import ProcessLog from "../ProcessLog";
import SwipeCompare from "../SwipeCompare";
import PipelineStrip from "../PipelineStrip";
import { stepsFor } from "../../processSteps";

export default function CompareSection({ city, cityLabel, years }) {
  const [yearA, setYearA] = useState(years[0]);
  const [yearB, setYearB] = useState(years[years.length - 1]);
  const [data, setData] = useState(null);
  const [overlays, setOverlays] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const run = () => {
    if (yearA === yearB) {
      setError("Choose two different years");
      return;
    }
    setError(null);
    setLoading(true);
    setData(null);
    Promise.all([api.compare(city, yearA, yearB), api.analyze(city, yearA), api.analyze(city, yearB)])
      .then(([cmp, a, b]) => {
        setData(cmp);
        setOverlays({ a: a.overlay_url, b: b.overlay_url });
      })
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  };

  useEffect(() => {
    run();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [city]);

  return (
    <section className="feature-section">
      <div className="section-eyebrow">Two-year comparison</div>
      <h2 className="section-title">What changed, where</h2>
      <PipelineStrip mode="compare" />

      <div className="controls inline-controls">
        <div className="ctrl-group">
          <span className="ctrl-label">From</span>
          <select value={yearA} onChange={(e) => setYearA(e.target.value)}>
            {years.map((y) => (
              <option key={y}>{y}</option>
            ))}
          </select>
          <span className="ctrl-label">To</span>
          <select value={yearB} onChange={(e) => setYearB(e.target.value)}>
            {years.map((y) => (
              <option key={y}>{y}</option>
            ))}
          </select>
        </div>
        <button className="run-btn" onClick={run} disabled={loading}>
          {loading ? "Comparing…" : "Compare"}
        </button>
        {error && <span className="status-msg error">{error}</span>}
      </div>

      {loading && (
        <div className="stage-loader embedded">
          <ProcessLog steps={stepsFor("compare", { city: cityLabel, yearA, yearB })} />
        </div>
      )}

      {data && overlays && !loading && (
        <>
          <p className="section-sub">
            Drag the slider. Same study area, {data.year_a} on the left, {data.year_b} on the
            right.
          </p>
          <div className="compare-slider-shell">
            <SwipeCompare
              imgA={bust(overlays.a)}
              imgB={bust(overlays.b)}
              labelA={`${cityLabel.toUpperCase()} · ${data.year_a}`}
              labelB={`${cityLabel.toUpperCase()} · ${data.year_b}`}
            />
          </div>

          <div className="scroll-cue">Scroll down for the change map &darr;</div>

          <div className="two-col diff-row">
            <div className="mask-viewer">
              <img className="mask-viewer-img" src={bust(data.diff_url)} alt="Change diff" />
              <div className="diff-legend static-legend">
                <span><span className="legend-dot" style={{ background: "#1a6b3a" }} />Stable forest</span>
                <span><span className="legend-dot" style={{ background: "#c0392b" }} />Lost</span>
                <span><span className="legend-dot" style={{ background: "#e8b400" }} />Gained</span>
              </div>
            </div>
            <div className="side-stats">
              <div className="big-stat">
                <div className={`big-stat-value ${data.change.net_ha < 0 ? "loss-color" : "gain-color"}`}>
                  {Math.abs(data.change.net_ha)}
                </div>
                <div className="big-stat-unit">hectares net {data.change.net_ha < 0 ? "lost" : "gained"}</div>
                <div className="big-stat-label">{data.year_a} &rarr; {data.year_b}</div>
              </div>
              <div className="change-row">
                <span className="change-key">Forest lost</span>
                <span className="change-val neg">{data.change.lost_ha} ha</span>
              </div>
              <div className="change-row">
                <span className="change-key">Forest gained</span>
                <span className="change-val pos">{data.change.gained_ha} ha</span>
              </div>
              <div className="change-row">
                <span className="change-key">Carbon released</span>
                <span className="change-val neg">{data.change.lost_carbon_t.toLocaleString("en-IN")} t-C</span>
              </div>
              <div className="change-row">
                <span className="change-key">CO&#8322; equivalent</span>
                <span className="change-val neg">{data.change.lost_co2_t.toLocaleString("en-IN")} t</span>
              </div>
              <div className="change-row">
                <span className="change-key">Minimum mapping unit</span>
                <span className="change-val">{data.min_mapping_unit_ha} ha</span>
              </div>
            </div>
          </div>
        </>
      )}
    </section>
  );
}
