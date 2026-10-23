// API client for the ForestWatch India FastAPI backend.
//
// Base URL is a single exported constant (not scattered through
// components) so switching to a deployed backend later is a one-line
// change -- matches the known "hardcoded API URL" roadmap item from the
// original index.html, made slightly easier to fix here.
export const API_BASE = import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";

// Dedupe identical concurrent GETs -- React StrictMode double-invokes
// every effect in development, which otherwise fires two parallel
// requests to the same slow endpoint (conservation-priority and risk
// both run a full CA-ANN pass, 20-30s each) and roughly doubles the real
// wait by making the backend do the same expensive work twice at once.
const inFlight = new Map();

async function getJSON(path) {
  if (inFlight.has(path)) return inFlight.get(path);

  const promise = (async () => {
    const res = await fetch(`${API_BASE}${path}`);
    if (!res.ok) {
      let detail = res.statusText;
      try {
        const body = await res.json();
        detail = body.detail || detail;
      } catch {
        /* response wasn't JSON, fall back to statusText */
      }
      throw new Error(detail);
    }
    return res.json();
  })();

  inFlight.set(path, promise);
  try {
    return await promise;
  } finally {
    inFlight.delete(path);
  }
}

// Direct URL to an already-fetched satellite image, bypassing /analyze
// entirely. Used by the timelapse scrubber: raw imagery for every year
// already exists on disk once satellite_fetch.py has run, so there's no
// need to call /analyze (which recomputes a mask + overlay) just to show
// the underlying composite for a given year.
export function satelliteUrl(city, year) {
  return `${API_BASE}/static/satellite_${city}_${year}.png`;
}

export const api = {
  cities: () => getJSON("/cities"),
  years: (city) => getJSON(`/years/${city}`),
  analyze: (city, year) => getJSON(`/analyze/${city}/${year}`),
  compare: (city, yearA, yearB) => getJSON(`/compare/${city}/${yearA}/${yearB}`),
  predict: (city, targetYear) => getJSON(`/predict/${city}/${targetYear}`),
  risk: (city) => getJSON(`/risk/${city}`),
  conservationPriority: (city, minPatchHa = 0.5, topN = 10) =>
    getJSON(`/conservation-priority/${city}?min_patch_ha=${minPatchHa}&top_n=${topN}`),
  validateHansen: (city, year, canopyThreshold = 10, bufferM = 30) =>
    getJSON(
      `/validate-hansen/${city}/${year}?canopy_threshold=${canopyThreshold}&buffer_m=${bufferM}`
    ),
};

// Appends a cache-busting query param -- mirrors the original
// index.html's `?t=${Date.now()}` pattern so re-running the same
// analysis after the backend regenerates an image doesn't show a
// stale cached PNG.
export function bust(url) {
  return `${url}?t=${Date.now()}`;
}
