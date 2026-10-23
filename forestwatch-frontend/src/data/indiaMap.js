// India outline, traced programmatically (OpenCV contour extraction +
// polygon simplification) from a real reference outline map rather than
// hand-plotted, so the silhouette is actually accurate. Coordinates are
// in the source image's own pixel space (800x944); city pins below are
// calibrated against the same space using two known lat/lon landmarks
// (Kanyakumari and Gujarat's westernmost point), verified against a
// third (Arunachal's easternmost tip, independent of the fit).
export const INDIA_VIEWBOX = "0 0 800 944";

export const INDIA_OUTLINE =
  "M174,53 L160,76 L191,98 L177,117 L181,154 L218,184 L135,298 L99,294 " +
  "L78,317 L111,394 L55,395 L38,411 L76,439 L56,451 L91,490 L148,478 " +
  "L165,622 L241,810 L274,842 L295,815 L325,812 L313,798 L338,782 " +
  "L347,633 L397,611 L471,527 L508,506 L517,472 L568,465 L554,393 " +
  "L538,387 L559,367 L538,343 L577,346 L593,367 L639,365 L616,408 " +
  "L627,425 L645,409 L655,449 L665,447 L708,318 L733,297 L755,300 " +
  "L759,268 L723,237 L689,244 L642,289 L618,294 L630,303 L620,320 " +
  "L558,322 L551,291 L534,293 L527,336 L403,311 L338,278 L357,238 " +
  "L303,195 L303,179 L324,168 L311,132 L336,85 L312,75 L272,91 L225,45 Z";

// lon -> x: x = 24.66*lon - 1639.0 ; lat -> y: y = -27.95*lat + 1067.8
// The four `active` cities have validated end-to-end pipelines and are
// clickable. The rest are shown for geographic context only.
export const CITIES = [
  { key: "delhi", label: "Delhi", x: 265, y: 268, active: true },
  { key: "kolkata", label: "Kolkata", x: 540, y: 437, active: true },
  { key: "bengaluru", label: "Bengaluru", x: 274, y: 705, active: true },
  { key: "chennai", label: "Chennai", x: 340, y: 702, active: true, labelBelow: true },
  { key: "mumbai", label: "Mumbai", x: 158, y: 535 },
  { key: "pune", label: "Pune", x: 182, y: 550 },
  { key: "hyderabad", label: "Hyderabad", x: 297, y: 582 },
  { key: "ahmedabad", label: "Ahmedabad", x: 151, y: 424 },
  { key: "jaipur", label: "Jaipur", x: 230, y: 316 },
  { key: "lucknow", label: "Lucknow", x: 357, y: 317 },
];
