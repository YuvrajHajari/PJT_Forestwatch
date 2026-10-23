// Curated real Chennai green-space/locality names with approximate
// coordinates, used to label a conservation-priority or risk patch with
// its nearest named place. Same "nearest named place" caveat as
// bengaluruAreas.js: approx_pixel_latlon (main.py) is a simple
// equirectangular estimate, not a survey-grade geocode, and this list is
// hand-curated to the project's own ~6km fetch radius around central
// Chennai, not a full gazetteer.
export const CHENNAI_AREAS = [
  { name: "Semmozhi Poonga", lat: 13.0444, lon: 80.2518 },
  { name: "Chetpet Eco Park", lat: 13.0689, lon: 80.2422 },
  { name: "Island Grounds", lat: 13.0677, lon: 80.2706 },
  { name: "People's Park", lat: 13.0837, lon: 80.2762 },
  { name: "Panagal Park", lat: 13.0412, lon: 80.2341 },
  { name: "Nageswara Rao Park", lat: 13.0367, lon: 80.2676 },
  { name: "Government Estate", lat: 13.0801, lon: 80.2871 },
  { name: "Nungambakkam", lat: 13.0569, lon: 80.2425 },
  { name: "Kilpauk", lat: 13.0775, lon: 80.2419 },
  { name: "Purasaiwalkam", lat: 13.0875, lon: 80.2565 },
  { name: "Egmore", lat: 13.0732, lon: 80.2609 },
  { name: "Triplicane", lat: 13.0569, lon: 80.2747 },
  { name: "Perambur", lat: 13.1134, lon: 80.2326 },
  { name: "Vepery", lat: 13.0875, lon: 80.2639 },
  { name: "Royapettah", lat: 13.0569, lon: 80.2637 },
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
  for (const area of CHENNAI_AREAS) {
    const d = haversineKm(lat, lon, area.lat, area.lon);
    if (d < bestDist) {
      bestDist = d;
      best = area;
    }
  }
  return best ? { name: best.name, distanceKm: Math.round(bestDist * 10) / 10 } : null;
}
