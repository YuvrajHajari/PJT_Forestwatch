import { useState } from "react";
import { bust } from "../api";
import SwipeCompare from "./SwipeCompare";
import TimelapseView from "./TimelapseView";

function Placeholder({ text }) {
  return (
    <div className="stage-placeholder">
      <svg width="48" height="48" viewBox="0 0 48 48" fill="none">
        <rect x="6" y="6" width="36" height="36" rx="3" stroke="#555" strokeWidth="1.5" />
        <path
          d="M6 30l10-10 8 8 8-12 10 14"
          stroke="#555"
          strokeWidth="1.5"
          strokeLinecap="round"
          strokeLinejoin="round"
          fill="none"
        />
      </svg>
      <span>{text}</span>
    </div>
  );
}

function Loader() {
  return (
    <div className="stage-loader">
      <div className="loader-inner">
        <div className="spinner" />
        <span>Analysing…</span>
      </div>
    </div>
  );
}

export default function Stage({
  mode,
  city,
  data,
  loading,
  satView,
  setSatView,
  priorityBackdrop,
}) {
  const empty = !data && !loading;
  const [compareView, setCompareView] = useState("diff"); // "diff" | "swipe"

  return (
    <div className="stage">
      {loading && <Loader />}

      {empty && <Placeholder text="Select a city and mode, then click Analyse" />}

      {!empty && mode === "single" && data && (
        <div style={{ height: "100%", position: "relative" }}>
          <div className="view-strip">
            {["satellite", "overlay", "mask"].map((v) => (
              <button
                key={v}
                className={`vbtn${satView === v ? " active" : ""}`}
                onClick={() => setSatView(v)}
              >
                {v}
              </button>
            ))}
          </div>
          <img
            className="stage-img"
            src={bust(data[`${satView}_url`])}
            alt={`${city} ${data.year} ${satView}`}
          />
          <span className="img-label left">
            {city.toUpperCase()} · {data.year}
          </span>
        </div>
      )}

      {!empty && mode === "compare" && data && (
        <div style={{ height: "100%", position: "relative" }}>
          <div className="swipe-view-toggle">
            {["diff", "swipe"].map((v) => (
              <button
                key={v}
                className={`vbtn${compareView === v ? " active" : ""}`}
                onClick={() => setCompareView(v)}
              >
                {v === "diff" ? "Diff map" : "Swipe"}
              </button>
            ))}
          </div>

          {compareView === "diff" ? (
            <>
              <div className="img-pair" style={{ height: "100%" }}>
                <div style={{ position: "relative" }}>
                  <img
                    className="stage-img"
                    src={bust(data.yearAOverlay)}
                    alt={`${city} ${data.year_a}`}
                  />
                  <span className="img-label left">
                    {city.toUpperCase()} · {data.year_a}
                  </span>
                </div>
                <div style={{ position: "relative" }}>
                  <img className="stage-img" src={bust(data.diff_url)} alt="Change diff" />
                  <span className="img-label right">
                    Change {data.year_a}→{data.year_b}
                  </span>
                </div>
              </div>
              <div className="diff-legend">
                <span>
                  <span className="legend-dot" style={{ background: "#1a6b3a" }} />
                  Stable forest
                </span>
                <span>
                  <span className="legend-dot" style={{ background: "#c0392b" }} />
                  Forest lost
                </span>
                <span>
                  <span className="legend-dot" style={{ background: "#e8b400" }} />
                  Forest gained
                </span>
              </div>
            </>
          ) : (
            <SwipeCompare
              imgA={bust(data.yearAOverlay)}
              imgB={bust(data.yearBOverlay)}
              labelA={`${city.toUpperCase()} · ${data.year_a}`}
              labelB={`${city.toUpperCase()} · ${data.year_b}`}
            />
          )}
        </div>
      )}

      {!empty && mode === "predict" && data && (
        <div style={{ height: "100%", position: "relative" }}>
          <img className="stage-img" src={bust(data.predict_url)} alt="Projection" />
          <span className="img-label left">
            {city.toUpperCase()} · PROJECTED {data.target_year} (FROM {data.base_year})
          </span>
          <div className="diff-legend">
            <span>
              <span className="legend-dot" style={{ background: "#1a6b3a" }} />
              Measured forest
            </span>
            <span>
              <span
                className="legend-hatch"
                style={{
                  borderColor: "#e8b400",
                  backgroundImage:
                    "repeating-linear-gradient(45deg,#e8b400 0,#e8b400 3px,transparent 3px,transparent 6px)",
                }}
              />
              Projected gain
            </span>
            <span>
              <span
                className="legend-hatch"
                style={{
                  borderColor: "#c0392b",
                  backgroundImage:
                    "repeating-linear-gradient(45deg,#c0392b 0,#c0392b 3px,transparent 3px,transparent 6px)",
                }}
              />
              Projected loss
            </span>
          </div>
        </div>
      )}

      {!empty && mode === "risk" && data && (
        <div style={{ height: "100%", position: "relative" }}>
          <img className="stage-img" src={bust(data.risk_heatmap_url)} alt="Risk heatmap" />
          <span className="img-label left">
            {city.toUpperCase()} · {data.year} · DEFORESTATION RISK
          </span>
          <div className="diff-legend">
            <span>
              <span className="legend-dot" style={{ background: "#2a3d8f" }} />
              Lower relative risk
            </span>
            <span>
              <span className="legend-dot" style={{ background: "#e8b400" }} />
              Moderate
            </span>
            <span>
              <span className="legend-dot" style={{ background: "#c0392b" }} />
              Higher relative risk
            </span>
          </div>
        </div>
      )}

      {!empty && mode === "conservation" && data && (
        <div style={{ height: "100%", position: "relative" }}>
          {priorityBackdrop && (
            <img className="stage-img" src={bust(priorityBackdrop)} alt={`${city} forest cover`} />
          )}
          {data.priority_patches?.map((p, i) => (
            <div
              key={p.patch_id}
              title={`#${i + 1} · ${p.area_ha} ha · risk ${p.mean_risk}`}
              style={{
                position: "absolute",
                left: `${(p.centroid_px.x / 1024) * 100}%`,
                top: `${(p.centroid_px.y / 1024) * 100}%`,
                transform: "translate(-50%, -50%)",
                width: 28,
                height: 28,
                borderRadius: "50%",
                background: "rgba(192,57,43,0.9)",
                border: "2px solid #f5f2ed",
                color: "#f5f2ed",
                fontFamily: "var(--mono)",
                fontSize: 12,
                fontWeight: 600,
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                boxShadow: "0 2px 6px rgba(0,0,0,0.4)",
              }}
            >
              {i + 1}
            </div>
          ))}
          <span className="img-label left">
            {city.toUpperCase()} · {data.year} · TOP {data.priority_patches?.length || 0} PRIORITY PATCHES
          </span>
        </div>
      )}

      {!empty && mode === "hansen" && data && (
        <div style={{ height: "100%", position: "relative" }}>
          <img className="stage-img" src={bust(data.diff_url)} alt="Hansen comparison" />
          <span className="img-label left">
            {city.toUpperCase()} · {data.year} · VS HANSEN GFC
          </span>
          <div className="diff-legend">
            <span>
              <span className="legend-dot" style={{ background: "#1a6b3a" }} />
              Both agree
            </span>
            <span>
              <span className="legend-dot" style={{ background: "#e8b400" }} />
              Ours only
            </span>
            <span>
              <span className="legend-dot" style={{ background: "#c0392b" }} />
              Hansen only
            </span>
          </div>
        </div>
      )}

      {!empty && mode === "timelapse" && data && (
        <TimelapseView city={city} years={data.years} />
      )}

      {!empty && mode === "citycompare" && data && (
        <div className="img-pair" style={{ height: "100%" }}>
          <div style={{ position: "relative" }}>
            <img className="stage-img" src={bust(data.cityA.overlay_url)} alt={data.cityA.key} />
            <span className="img-label left">
              {data.cityA.key.toUpperCase()} · {data.cityA.year}
            </span>
          </div>
          <div style={{ position: "relative" }}>
            <img className="stage-img" src={bust(data.cityB.overlay_url)} alt={data.cityB.key} />
            <span className="img-label right">
              {data.cityB.key.toUpperCase()} · {data.cityB.year}
            </span>
          </div>
        </div>
      )}
    </div>
  );
}
