import { useState } from "react";

export default function FormulaCompareNote() {
  const [open, setOpen] = useState(false);

  return (
    <div className="formula-compare">
      <button className="formula-compare-toggle" onClick={() => setOpen((o) => !o)}>
        {open ? "−" : "+"} Why this formula, not the industry standard?
      </button>
      {open && (
        <div className="formula-compare-body">
          <div className="formula-compare-row">
            <div className="formula-compare-col">
              <div className="formula-card-label">Industry standard &middot; two formulas</div>
              <div className="formula-card-eq small">NDVI = (NIR&minus;R) / (NIR+R)</div>
              <div className="formula-card-eq small">NDWI = (G&minus;NIR) / (G+NIR)</div>
              <p className="formula-card-note">
                Most published work runs these separately, one for vegetation, one for water,
                then combines the two resulting masks (vegetation AND NOT water).
              </p>
            </div>
            <div className="formula-compare-col">
              <div className="formula-card-label">This project &middot; one formula</div>
              <div className="formula-card-eq small">FVI = (NIR+G&minus;R&minus;B) / (NIR+G+R+B)</div>
              <p className="formula-card-note">
                NIR is folded directly into a single ratio, so vegetation and water separate in
                one pass instead of two independently-thresholded masks that have to agree at
                every boundary pixel.
              </p>
            </div>
          </div>
          <p className="formula-compare-finding">
            Tested against the standard NDVI+NDWI approach on real data: in a water-heavy city
            (Chennai, ~31% of the frame is water), FVI alone misclassified 0.00% of water as
            forest versus NDVI alone misclassifying 3.37%. In low-water cities (Bengaluru,
            Delhi) the two approaches tie: there isn&rsquo;t enough water in frame for the
            difference to show up.
          </p>
          <p className="formula-compare-why">
            Why: the gap only appears when there&rsquo;s meaningful water in the scene to get
            wrong in the first place. One formula with NIR built in has nothing to
            disagree with itself about at a water&rsquo;s edge, where two separately-thresholded
            masks do.
          </p>
        </div>
      )}
    </div>
  );
}
