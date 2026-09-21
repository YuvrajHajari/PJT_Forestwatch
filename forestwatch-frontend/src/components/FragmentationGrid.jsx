// Shared fragmentation display -- used by both SinglePanel and
// PredictPanel (base year + projected year), so both surfaces stay in
// sync automatically if the field set ever changes.
export default function FragmentationGrid({ fragmentation }) {
  if (!fragmentation) return null;
  return (
    <div className="frag-grid">
      <div className="frag-cell">
        <div className="frag-cell-label">Patch count</div>
        <div className="frag-cell-value">{fragmentation.patch_count}</div>
        <div className="frag-cell-unit">distinct patches</div>
      </div>
      <div className="frag-cell">
        <div className="frag-cell-label">Mean patch size</div>
        <div className="frag-cell-value">{fragmentation.mean_patch_size_ha}</div>
        <div className="frag-cell-unit">ha</div>
      </div>
      <div className="frag-cell">
        <div className="frag-cell-label">Largest patch</div>
        <div className="frag-cell-value">{fragmentation.largest_patch_index_pct}%</div>
        <div className="frag-cell-unit">of total forest</div>
      </div>
      <div className="frag-cell">
        <div className="frag-cell-label">Edge density</div>
        <div className="frag-cell-value">{fragmentation.edge_density_m_per_ha}</div>
        <div className="frag-cell-unit">m / ha</div>
      </div>
    </div>
  );
}
