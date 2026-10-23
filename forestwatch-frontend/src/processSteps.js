// Narration scripts for the process log shown while each mode's backend
// call is in flight. Step order mirrors the real pipeline order in
// main.py (not a generic "loading" placeholder) so the log is a true,
// if simplified, account of what the request is actually doing.

export function stepsFor(mode, ctx) {
  const city = (ctx.city || "").toUpperCase();

  switch (mode) {
    case "single":
      return [
        `Requesting Sentinel-2 scene for ${city} ${ctx.year}`,
        "Decoding surface reflectance bands (B2 B3 B4 B8)",
        "Computing FVI = (NIR+G−R−B) / (NIR+G+R+B)",
        "Otsu three-class threshold, water / built-up / forest",
        "Morphological cleanup and patch labelling",
        "Measuring carbon stock and fragmentation",
      ];
    case "compare":
      return [
        `Requesting two scenes for ${city}, ${ctx.yearA} and ${ctx.yearB}`,
        "Running per-image FVI + Otsu on each year independently",
        "Differencing the two masks, pixel by pixel",
        "Filtering boundary noise below the 0.1 ha mapping unit",
        "Compositing the three-colour change map",
      ];
    case "predict":
      return [
        `Loading ${city}'s historical mask series`,
        "Re-thresholding all years at one pooled cutoff (removes drift)",
        "Estimating the Markov transition rate from gaps ≥3 years apart",
        "Scoring candidate pixels with the CA-ANN ensemble (8 seeds)",
        "Calibrating probabilities for class imbalance",
        "Ranking pixels and spending the learned rate budget",
        `Compositing the ${ctx.targetYear} projection overlay`,
      ];
    case "risk":
      return [
        `Loading ${city}'s latest mask and trained CA-ANN ensemble`,
        "Scoring every forest pixel's change probability",
        "Ranking scores by percentile (relative, not absolute, risk)",
        "Rendering the heatmap over a dimmed satellite backdrop",
      ];
    case "conservation":
      return [
        `Scoring per-pixel deforestation risk for ${city}`,
        "Labelling connected forest patches",
        "Discarding patches under 0.5 ha",
        "Ranking by priority = normalised area × mean risk",
        "Fetching a satellite backdrop for patch markers",
      ];
    case "hansen":
      return [
        `Loading Hansen Global Forest Change rasters for ${city}`,
        `Reconstructing Hansen's implied ${ctx.hansenYear} forest mask`,
        "Comparing against our FVI mask, pixel by pixel",
        "Computing the strict confusion matrix (precision/recall/Kappa)",
        "Recomputing agreement with a 30 m positional tolerance",
      ];
    case "citycompare":
      return [
        `Requesting scenes for ${city} and ${(ctx.compareCityB || "").toUpperCase()}`,
        "Running FVI + Otsu independently for each city",
        "Aligning fragmentation and carbon metrics side by side",
      ];
    case "timelapse":
      return [`Indexing fetched imagery years for ${city}`];
    default:
      return ["Working…"];
  }
}
