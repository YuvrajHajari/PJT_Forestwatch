import { cap } from "../../narrative";

export default function CityComparePanel({ data }) {
  const { cityA, cityB } = data;

  const rows = [
    { label: "Forest area (ha)", a: cityA.forest_area_ha, b: cityB.forest_area_ha, higherBetter: true },
    { label: "Coverage (%)", a: cityA.coverage_pct, b: cityB.coverage_pct, higherBetter: true },
    { label: "Carbon stock (t-C)", a: cityA.carbon_t, b: cityB.carbon_t, higherBetter: true },
    {
      label: "Patch count",
      a: cityA.fragmentation?.patch_count,
      b: cityB.fragmentation?.patch_count,
      higherBetter: false,
    },
    {
      label: "Mean patch size (ha)",
      a: cityA.fragmentation?.mean_patch_size_ha,
      b: cityB.fragmentation?.mean_patch_size_ha,
      higherBetter: true,
    },
    {
      label: "Largest patch (%)",
      a: cityA.fragmentation?.largest_patch_index_pct,
      b: cityB.fragmentation?.largest_patch_index_pct,
      higherBetter: true,
    },
    {
      label: "Edge density (m/ha)",
      a: cityA.fragmentation?.edge_density_m_per_ha,
      b: cityB.fragmentation?.edge_density_m_per_ha,
      higherBetter: false,
    },
  ];

  return (
    <>
      <div className="panel-section">
        <div className="section-eyebrow">
          {cap(cityA.key)} vs {cap(cityB.key)} · {cityA.year}
        </div>
        <table className="citycompare-table">
          <thead>
            <tr>
              <th>Metric</th>
              <th className="num">{cap(cityA.key)}</th>
              <th className="num">{cap(cityB.key)}</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => {
              if (r.a == null || r.b == null) return null;
              const aWins = r.higherBetter ? r.a > r.b : r.a < r.b;
              const bWins = r.higherBetter ? r.b > r.a : r.b < r.a;
              return (
                <tr key={r.label}>
                  <td className="label">{r.label}</td>
                  <td className={`num${aWins ? " citycompare-winner" : ""}`}>{r.a}</td>
                  <td className={`num${bWins ? " citycompare-winner" : ""}`}>{r.b}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
        <p style={{ fontSize: 11, color: "var(--ink3)", marginTop: 12, lineHeight: 1.6 }}>
          Green highlights the stronger value per metric (more forest, more carbon, less
          fragmentation).
        </p>
      </div>
    </>
  );
}
