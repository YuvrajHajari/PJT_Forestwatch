import sys
sys.path.insert(0, '.')
import main as fw

# Load all masks
city = 'chennai'
years = ['2016', '2017', '2018', '2019', '2020', '2021', '2022', '2023', '2024', '2025']

masks_by_year = {}
for y in years:
    try:
        img = fw.load_satellite(city, y)
        nir = fw.load_nir(city, y)
        mask, (t1, t2) = fw.compute_fvi_mask(img, nir)
        masks_by_year[y] = mask > 127
        
        # Compute area
        pixel_ha = fw.pixel_area_ha(fw.CITIES[city]["buffer_km"])
        forest_px = (mask > 127).sum()
        area_ha = forest_px * pixel_ha
        
        print(f"{y}: {area_ha:.1f} ha  |  FVI thresholds: t1={t1:.4f}  t2={t2:.4f}")
    except Exception as e:
        print(f"{y}: ERROR - {e}")

# Compute rates
p_gain, p_loss = fw.compute_annual_transition_rates(masks_by_year)
print(f"\nMarkov rates: p_gain={p_gain*100:.3f}%/yr  |  p_loss={p_loss*100:.3f}%/yr")