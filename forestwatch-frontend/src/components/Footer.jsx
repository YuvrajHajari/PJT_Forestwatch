export default function Footer() {
  return (
    <footer className="footer">
      <span>
        <strong>Method:</strong> FVI = (NIR+Green−Red−Blue)/(NIR+Green+Red+Blue) ·
        multi-level Otsu (per-image, adaptive)
      </span>
      <span>
        <strong>Carbon:</strong> 190 t-C/ha (IPCC tropical moist forest) · CO₂ ×3.67
      </span>
      <span>
        <strong>Source:</strong> Copernicus Sentinel-2 via Google Earth Engine
      </span>
    </footer>
  );
}
