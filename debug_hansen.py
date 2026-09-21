"""
debug_hansen.py -- Diagnostic for the Hansen validation anomaly.

hansen_forest_area_ha came back as only 10.16 ha for Chennai 2025 (vs
our_forest_area_ha=872.48) -- far beyond what the documented gain-band
limitation alone could explain. This inspects the raw fetched Hansen
arrays directly, before any thresholding, to localize whether the
problem is in the GEE fetch itself or in the Python-side decoding.

Usage:
    python debug_hansen.py
"""
import sys
import numpy as np
sys.path.insert(0, ".")
import main as fw

for city in fw.CITIES:
    print(f"\n{'='*60}\n{city}\n{'='*60}")

    treecover = fw.load_hansen_treecover(city)  # should be 0-100
    lossyear  = fw.load_hansen_lossyear(city)   # should be 0-25

    print(f"\n--- treecover2000 ---")
    print(f"shape: {treecover.shape}, dtype: {treecover.dtype}")
    print(f"min: {treecover.min():.2f}  max: {treecover.max():.2f}  mean: {treecover.mean():.2f}")
    for p in [10, 25, 50, 75, 90, 95, 99]:
        print(f"  p{p}: {np.percentile(treecover, p):.2f}")
    print(f"pixels >= 30%: {int(np.sum(treecover >= 30))} / {treecover.size} "
          f"({np.mean(treecover >= 30)*100:.4f}%)")
    print(f"pixels >= 10%: {int(np.sum(treecover >= 10))} / {treecover.size} "
          f"({np.mean(treecover >= 10)*100:.4f}%)")
    print(f"pixels == 0:   {int(np.sum(treecover == 0))} / {treecover.size} "
          f"({np.mean(treecover == 0)*100:.4f}%)")

    print(f"\n--- lossyear ---")
    print(f"shape: {lossyear.shape}, dtype: {lossyear.dtype}")
    print(f"min: {lossyear.min():.2f}  max: {lossyear.max():.2f}  mean: {lossyear.mean():.2f}")
    ly_round = np.round(lossyear).astype(int)
    print(f"pixels == 0 (no loss recorded): {int(np.sum(ly_round == 0))} / {lossyear.size} "
          f"({np.mean(ly_round == 0)*100:.4f}%)")

    pixel_ha = fw.pixel_area_ha(fw.CITIES[city]["buffer_km"])

    print(f"\n--- reconstruction ---")
    forest_2000 = treecover >= 30
    print(f"forest_2000 (treecover>=30%, no loss filter yet): "
          f"{int(np.sum(forest_2000)) * pixel_ha:.2f} ha")

    hansen_2025 = fw.compute_hansen_forest_mask(treecover, lossyear, 2025, 30)
    print(f"forest_2025 (after loss filter):                  "
          f"{int(np.sum(hansen_2025)) * pixel_ha:.2f} ha")

    # Also check the raw PNG files directly, bypassing load_hansen_* entirely,
    # to see if the resize/decode step in main.py is doing anything unexpected.
    import cv2, os
    raw_tc_path = os.path.join(fw.STATIC_DIR, f"hansen_treecover_{city}.png")
    raw_tc = cv2.imread(raw_tc_path, cv2.IMREAD_UNCHANGED)
    print(f"\n--- raw PNG file (bypassing load_hansen_treecover) ---")
    print(f"file: {raw_tc_path}")
    print(f"raw shape: {raw_tc.shape if raw_tc is not None else 'FAILED TO LOAD'}, "
          f"dtype: {raw_tc.dtype if raw_tc is not None else 'n/a'}")
    if raw_tc is not None:
        print(f"raw min: {raw_tc.min()}  max: {raw_tc.max()}  mean: {raw_tc.mean():.2f}")
        if raw_tc.ndim == 3:
            print(f"channels: {raw_tc.shape[2]} (if 4, likely has an alpha channel)")