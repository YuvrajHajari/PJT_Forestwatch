import { useEffect, useState } from "react";
import { api, bust } from "../../api";
import ProcessLog from "../ProcessLog";
import PipelineStrip from "../PipelineStrip";
import { stepsFor } from "../../processSteps";
import { nearestArea as nearestBengaluruArea } from "../../data/bengaluruAreas";
import { nearestArea as nearestChennaiArea } from "../../data/chennaiAreas";

const AREA_LOOKUP = { bengaluru: nearestBengaluruArea, chennai: nearestChennaiArea };

function MethodNote({ cv, ml, children }) {
  return (
    <div className="method-note">
      <div className="method-note-tags">
        {cv && <span className="method-tag cv">computer vision</span>}
        {ml && <span className="method-tag ml">machine learning</span>}
      </div>
      <p>{children}</p>
    </div>
  );
}

function RiskView({ city, cityLabel }) {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    setLoading(true);
    api.risk(city).then(setData).finally(() => setLoading(false));
  }, [city]);

  return (
    <div className="more-feature-body">
      <MethodNote cv ml>
        The trained network scores every forest pixel's change probability (ML). That raw score
        is then ranked by percentile and painted with a colour ramp (CV), because raw
        probabilities cluster too low for a straight colour mapping to show anything.
      </MethodNote>
      <PipelineStrip mode="risk" />
      {loading && (
        <div className="stage-loader embedded">
          <ProcessLog steps={stepsFor("risk", { city: cityLabel })} />
        </div>
      )}
      {data && !loading && (
        <div className="two-col diff-row">
          <div className="mask-viewer">
            <img className="mask-viewer-img" src={bust(data.risk_heatmap_url)} alt="Risk heatmap" />
            <div className="img-label left">{cityLabel.toUpperCase()} &middot; {data.year}</div>
            <div className="diff-legend static-legend">
              <span><span className="legend-dot" style={{ background: "#2a3d8f" }} />Lower risk</span>
              <span><span className="legend-dot" style={{ background: "#e8b400" }} />Moderate</span>
              <span><span className="legend-dot" style={{ background: "#c0392b" }} />Higher risk</span>
            </div>
          </div>
          <div className="side-stats">
            <div className="big-stat">
              <div className="big-stat-value">{data.high_risk_area_ha}</div>
              <div className="big-stat-unit">ha in the top {100 - data.high_risk_percentile}% risk band</div>
            </div>
            <div className="change-row">
              <span className="change-key">Significant forest area</span>
              <span className="change-val">{data.significant_forest_area_ha} ha</span>
            </div>
            <div className="change-row">
              <span className="change-key">High-risk share of forest</span>
              <span className="change-val neg">{data.high_risk_pct_of_forest}%</span>
            </div>
            <div className="change-row">
              <span className="change-key">Mean relative risk</span>
              <span className="change-val">{data.mean_relative_risk_score}</span>
            </div>
            {AREA_LOOKUP[city] && (
              <p className="area-note">
                See Conservation Priority for the specific named areas carrying the highest risk.
              </p>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

function ConservationView({ city, cityLabel }) {
  const [data, setData] = useState(null);
  const [backdrop, setBackdrop] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    setLoading(true);
    Promise.all([api.conservationPriority(city), api.years(city)])
      .then(([priority, yearsRes]) => {
        setData(priority);
        const latest = yearsRes.years[yearsRes.years.length - 1];
        return api.analyze(city, latest).then((a) => setBackdrop(a.overlay_url));
      })
      .finally(() => setLoading(false));
  }, [city]);

  return (
    <div className="more-feature-body">
      <MethodNote cv ml>
        OpenCV's connected-components finds each forest patch (CV). Each patch's priority score
        is its normalised area multiplied by the network's mean risk score within it (ML), so a
        patch ranks highest when it is both large and at risk.
      </MethodNote>
      <PipelineStrip mode="conservation" />
      {loading && (
        <div className="stage-loader embedded">
          <ProcessLog steps={stepsFor("conservation", { city: cityLabel })} />
        </div>
      )}
      {data && !loading && (
        <div className="two-col diff-row">
          <div className="mask-viewer">
            {backdrop && <img className="mask-viewer-img" src={bust(backdrop)} alt="backdrop" />}
            {data.priority_patches.map((p, i) => (
              <div
                key={p.patch_id}
                className="priority-pin"
                style={{ left: `${(p.centroid_px.x / 1024) * 100}%`, top: `${(p.centroid_px.y / 1024) * 100}%` }}
                title={`#${i + 1}`}
              >
                {i + 1}
              </div>
            ))}
            <div className="img-label left">{cityLabel.toUpperCase()} &middot; {data.year}</div>
          </div>
          <div className="priority-list">
            {data.priority_patches.map((p, i) => {
              const area = AREA_LOOKUP[city] ? AREA_LOOKUP[city](p.approx_lat, p.approx_lon) : null;
              return (
                <div className="priority-row" key={p.patch_id}>
                  <div className="priority-rank">{i + 1}</div>
                  <div className="priority-body">
                    <div className="priority-top-row">
                      <span className="priority-area">{p.area_ha} ha</span>
                      <span className="priority-score">score {p.priority_score}</span>
                    </div>
                    {area && <div className="area-tag">near {area.name} (~{area.distanceKm} km)</div>}
                    <div className="priority-risk-bar">
                      <div className="priority-risk-fill" style={{ width: `${p.mean_risk * 100}%` }} />
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}

function CityCompareView({ city, cityLabel, years, cities, compareCityB, setCompareCityB }) {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const year = years[years.length - 1];

  useEffect(() => {
    if (!compareCityB) return;
    setLoading(true);
    Promise.all([api.analyze(city, year), api.analyze(compareCityB, year)])
      .then(([a, b]) => setData({ a, b }))
      .finally(() => setLoading(false));
  }, [city, compareCityB, year]);

  const rows = data
    ? [
        ["Forest area", `${data.a.forest_area_ha} ha`, `${data.b.forest_area_ha} ha`, data.a.forest_area_ha > data.b.forest_area_ha],
        ["Coverage", `${data.a.coverage_pct}%`, `${data.b.coverage_pct}%`, data.a.coverage_pct > data.b.coverage_pct],
        ["Carbon stock", `${data.a.carbon_t.toLocaleString("en-IN")} t`, `${data.b.carbon_t.toLocaleString("en-IN")} t`, data.a.carbon_t > data.b.carbon_t],
        ["Patch count", data.a.fragmentation.patch_count, data.b.fragmentation.patch_count, data.a.fragmentation.patch_count < data.b.fragmentation.patch_count],
        ["Mean patch size", `${data.a.fragmentation.mean_patch_size_ha} ha`, `${data.b.fragmentation.mean_patch_size_ha} ha`, data.a.fragmentation.mean_patch_size_ha > data.b.fragmentation.mean_patch_size_ha],
      ]
    : [];

  return (
    <div className="more-feature-body">
      <MethodNote cv>
        Pure computer vision, no model involved: the identical FVI + Otsu detection run
        independently for each city, then placed side by side.
      </MethodNote>
      <PipelineStrip mode="citycompare" />
      <div className="controls inline-controls">
        <span className="ctrl-label">Compare against</span>
        <select value={compareCityB} onChange={(e) => setCompareCityB(e.target.value)}>
          {cities.filter((c) => c.key !== city).map((c) => (
            <option key={c.key} value={c.key}>{c.label}</option>
          ))}
        </select>
      </div>
      {loading && <div className="stage-loader embedded"><ProcessLog steps={stepsFor("citycompare", { city: cityLabel, compareCityB })} /></div>}
      {data && !loading && (
        <>
          <div className="img-pair compare-pair">
            <div className="mask-viewer small">
              <img className="mask-viewer-img" src={bust(data.a.overlay_url)} alt={city} />
              <div className="img-label left">{cityLabel.toUpperCase()}</div>
            </div>
            <div className="mask-viewer small">
              <img className="mask-viewer-img" src={bust(data.b.overlay_url)} alt={compareCityB} />
              <div className="img-label left">{compareCityB.toUpperCase()}</div>
            </div>
          </div>
          <table className="citycompare-table">
            <thead>
              <tr><th></th><th className="num">{cityLabel}</th><th className="num">{compareCityB}</th></tr>
            </thead>
            <tbody>
              {rows.map(([label, a, b, aWins]) => (
                <tr key={label}>
                  <td className="label">{label}</td>
                  <td className={`num${aWins ? " citycompare-winner" : ""}`}>{a}</td>
                  <td className={`num${!aWins ? " citycompare-winner" : ""}`}>{b}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </>
      )}
    </div>
  );
}

export default function MoreFeaturesSection({ sub, city, cityLabel, years, cities, compareCityB, setCompareCityB }) {
  const titles = {
    risk: ["Risk heatmap", "Where change is most likely next"],
    conservation: ["Conservation priority", "Which patches matter most, ranked"],
    citycompare: ["City vs city", "Same method, two cities, side by side"],
  };
  const [eyebrow, title] = titles[sub] || ["", ""];

  return (
    <section className="feature-section">
      <div className="section-eyebrow">{eyebrow}</div>
      <h2 className="section-title">{title}</h2>
      {sub === "risk" && <RiskView city={city} cityLabel={cityLabel} />}
      {sub === "conservation" && <ConservationView city={city} cityLabel={cityLabel} />}
      {sub === "citycompare" && (
        <CityCompareView
          city={city}
          cityLabel={cityLabel}
          years={years}
          cities={cities}
          compareCityB={compareCityB}
          setCompareCityB={setCompareCityB}
        />
      )}
    </section>
  );
}
