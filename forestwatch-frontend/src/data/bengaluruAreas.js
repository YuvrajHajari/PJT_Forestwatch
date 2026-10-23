// Curated real Bengaluru green-space/locality names with approximate
// coordinates, used to label a conservation-priority or risk patch with
// its nearest named place. "Nearest named place" is an honest framing --
// the backend's approx_pixel_latlon is a simple equirectangular estimate
// (main.py), not a survey-grade geocode, and this list is hand-curated,
// not a full gazetteer. Good enough to make a ranked list legible to a
// non-technical viewer rather than a bare lat/lon pair.
export const BENGALURU_AREAS = [
  { name: "Cubbon Park", lat: 12.9763, lon: 77.5929 },
  { name: "Lalbagh Botanical Garden", lat: 12.9507, lon: 77.5848 },
  { name: "Turahalli Forest", lat: 12.8783, lon: 77.5367 },
  { name: "Hebbal Lake", lat: 13.045, lon: 77.595 },
  { name: "Jayamahal", lat: 12.9975, lon: 77.597 },
  { name: "Hesaraghatta", lat: 13.1352, lon: 77.4828 },
  { name: "JP Nagar", lat: 12.9077, lon: 77.5851 },
  { name: "Yelahanka", lat: 13.1005, lon: 77.5963 },
  { name: "Bellandur Lake", lat: 12.9304, lon: 77.6784 },
  { name: "Nagarbhavi", lat: 12.9581, lon: 77.5017 },
  { name: "Puttenahalli Lake", lat: 12.8917, lon: 77.5827 },
  { name: "Kaikondrahalli Lake", lat: 12.9123, lon: 77.6762 },
  { name: "Jarakabande Kaval", lat: 13.0706, lon: 77.5287 },
  { name: "Sankey Tank", lat: 13.007, lon: 77.573 },
  { name: "Madiwala Lake", lat: 12.9218, lon: 77.6192 },
  { name: "Bannerghatta Road", lat: 12.8, lon: 77.58 },
  { name: "Whitefield", lat: 12.9698, lon: 77.75 },
  { name: "Doddanekundi", lat: 12.975, lon: 77.695 },
];

function haversineKm(lat1, lon1, lat2, lon2) {
  const R = 6371;
  const dLat = ((lat2 - lat1) * Math.PI) / 180;
  const dLon = ((lon2 - lon1) * Math.PI) / 180;
  const a =
    Math.sin(dLat / 2) ** 2 +
    Math.cos((lat1 * Math.PI) / 180) * Math.cos((lat2 * Math.PI) / 180) * Math.sin(dLon / 2) ** 2;
  return R * 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
}

export function nearestArea(lat, lon) {
  let best = null;
  let bestDist = Infinity;
  for (const area of BENGALURU_AREAS) {
    const d = haversineKm(lat, lon, area.lat, area.lon);
    if (d < bestDist) {
      bestDist = d;
      best = area;
    }
  }
  return best ? { name: best.name, distanceKm: Math.round(bestDist * 10) / 10 } : null;
}
