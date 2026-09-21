// Auto-generated narrative summaries. Deliberately pure template logic
// over data the app already has -- no new API calls, no LLM, no new
// dependency. The goal is synthesis, not decoration: turn several
// separately-displayed numbers into one sentence a reader can act on.

export function cap(s) {
  return s ? s.charAt(0).toUpperCase() + s.slice(1) : s;
}

function describeFragmentation(frag) {
  if (!frag || frag.patch_count === 0) {
    return "no forest patches were detected above the noise threshold";
  }
  if (frag.largest_patch_index_pct >= 30) {
    return (
      `a relatively consolidated distribution: the single largest patch alone accounts for ` +
      `${frag.largest_patch_index_pct}% of all detected forest`
    );
  }
  if (frag.largest_patch_index_pct >= 10) {
    return (
      `a moderately fragmented distribution across ${frag.patch_count} patches, with no single ` +
      `block dominating`
    );
  }
  return (
    `a highly fragmented distribution across ${frag.patch_count} small patches (mean size ` +
    `${frag.mean_patch_size_ha} ha), rather than large contiguous blocks`
  );
}

// Single-year analysis: current state + fragmentation character, plus a
// trend sentence if this city has been analysed at another year before
// (trendData comes from App's running per-city history, so this degrades
// gracefully to a single-year-only summary the first time a city is used).
export function singleYearNarrative(data, trendData) {
  const { city, year, forest_area_ha, coverage_pct, fragmentation } = data;
  const cityName = cap(city);

  let trendSentence = "";
  const otherYears = Object.keys(trendData || {})
    .filter((y) => y !== year)
    .sort();
  if (otherYears.length > 0) {
    const earliest = otherYears[0];
    const earliestVal = trendData[earliest];
    const diff = Math.round((forest_area_ha - earliestVal) * 100) / 100;
    const pct = earliestVal > 0 ? Math.round((diff / earliestVal) * 1000) / 10 : 0;
    const direction = diff >= 0 ? "grown" : "shrunk";
    trendSentence = ` Compared to ${earliest}, this has ${direction} by ${Math.abs(diff)} ha (${Math.abs(
      pct
    )}%).`;
  }

  return (
    `As of ${year}, ${cityName} has ${forest_area_ha.toLocaleString("en-IN")} hectares of detected ` +
    `forest cover, ${coverage_pct}% of the study area.${trendSentence} The forest shows ` +
    `${describeFragmentation(fragmentation)}.`
  );
}

// Prediction: net change + fragmentation trend (base year vs. projected
// year), both already computed by the backend for every /predict call.
export function predictionNarrative(data) {
  const { city, base_year, target_year, base_year_data, projected_data } = data;
  const cityName = cap(city);
  const diff = Math.round((projected_data.forest_area_ha - base_year_data.forest_area_ha) * 100) / 100;
  const direction = diff >= 0 ? "gain" : "loss";

  let fragTrend = "";
  const baseFrag = base_year_data.fragmentation;
  const projFrag = projected_data.fragmentation;
  if (baseFrag && projFrag && baseFrag.patch_count > 0) {
    const patchDelta = projFrag.patch_count - baseFrag.patch_count;
    if (patchDelta > 0) {
      fragTrend = ` Fragmentation is projected to increase, with patch count rising from ${baseFrag.patch_count} to ${projFrag.patch_count}.`;
    } else if (patchDelta < 0) {
      fragTrend = ` Fragmentation is projected to ease, with patch count falling from ${baseFrag.patch_count} to ${projFrag.patch_count}.`;
    } else {
      fragTrend = ` Patch count is projected to stay roughly flat at ${projFrag.patch_count}.`;
    }
  }

  return (
    `By ${target_year}, ${cityName}'s forest cover is projected to see a net ${direction} of ` +
    `${Math.abs(diff)} ha relative to ${base_year}.${fragTrend}`
  );
}
