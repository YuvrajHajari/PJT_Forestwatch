ive just switched from a regular claude communication on the projects tab to this vscode extension thing so you can work directly, weve been working on a project, but i dont think you know anything about that. still heres something for reference. 


PROJECT: ForestWatch India — final-year engineering thesis. FastAPI + OpenCV/sklearn
backend, React frontend, Google Earth Engine (Sentinel-2) data source. Two cities live
(Chennai, Bengaluru), Mumbai/Delhi/Kolkata scaffolded in cities_config.py but not yet
fetched. Repo: github.com/YuvrajHajari/PJT_Forestwatch (fresh history, git was just
cleaned — static/ and node_modules/ are gitignored now, never commit generated imagery).

=== WHAT'S ALREADY BUILT AND WORKING (do not rebuild, only extend) ===

Detection: FVI = (NIR+G-R-B)/(NIR+G+R+B), replacing an original VARI+NDWI approach
after VARI misclassified Chennai's coastal water as forest. Three-class Otsu threshold
splits water/land/vegetation in one pass. Per-image threshold for single-year analysis;
a GLOBAL threshold pooled across all years is used specifically for the CA-ANN pipeline
(per-image thresholds were found to drift up to 51% of the FVI scale across 10 years,
inflating apparent change).

Prediction (CA-ANN): two independent signals, combined only at final selection —
(1) HOW MUCH changes: historical Markov transition rate, computed only from year-pairs
≥3 years apart (consecutive-year pairs inflated Chennai's rate to an implausible
18.8%/yr from seasonal oscillation noise — real trend is 607.5ha 2016 → 872.5ha 2025).
(2) WHICH pixels change: MLPClassifier (scikit-learn), 4 pixel features (density @5x5,
15x15, 31x31 + edge distance) → 16 ReLU → 8 ReLU → 1 sigmoid, trained on pooled
consecutive-year transitions across both cities, calibrated post-hoc for negative-class
undersampling. An earlier independent-threshold design (no shared budget) was replaced
after it produced a symmetric ring artifact and unrealistic long-horizon erosion.

Validated results (walk-forward, current corrected pipeline): CA-ANN reaches near-parity
with a density-heuristic baseline at every horizon tested (1yr: Kappa 0.6988 vs 0.7028,
3yr: 0.5103 vs 0.5154, 6yr: 0.2164 vs 0.2149) — NOT a clean win. This is reported
honestly as a data-volume-ceiling finding (only 18 independent transitions currently),
not hidden. A 5th "density trend" feature was tested and reverted after it hurt
performance (1/8 wins vs 3/8) — also documented as a negative result.

Other features, all real and working: conservation priority ranking (priority_score =
normalized_patch_area × mean_CA-ANN_risk), a continuous deforestation risk heatmap
(contrast-stretched CA-ANN score, MMU-filtered, Gaussian-blurred), Hansen Global Forest
Change external validation (strict pixel-exact AND buffered/positional-tolerance metrics
— strict precision was as low as 0.037, buffered at 30m rose to 0.222, confirmed as a
genuine granularity mismatch not a bug, since agreement improves monotonically 30m→60m),
and FRAGSTATS-style fragmentation metrics (patch count, mean size, largest-patch index,
edge density). React frontend has 8 modes: single-year, compare, predict, risk heatmap,
conservation priority, Hansen validation, timelapse, city-vs-city.

Study area: 6km-radius buffer per city → 12km×12km bounding square = 144 km² actually
analyzed — roughly 20-35% of each city's official core-city limits, NOT the full city.
Be precise about this if asked.

=== REVIEW 2 DELIVERABLES (already done, both delivered as files) ===

1. A "simple, human, minimal" PPT (deliberately NOT icon-heavy/AI-styled) — recap,
   CA-ANN explanation, features, real screenshots.
2. A follow-up PPT addressing specific faculty pushback: why 20 images is enough
   (18.87M pixel-level training rows: 18 transition-pairs × ~1.05M pixels each — the
   real constraint is 18 independent transitions, not pixel count), why sub-annual
   slicing would backfire (seasonal noise, same 18.8% oscillation finding), a scaling
   table (current 20 city-years → +3 cities = 50 → target 100-150), both architecture
   diagrams, and the Hansen granularity argument with real screenshots.
3. A 15-paper IEEE-format literature review paper, true 2-column layout, real verified
   citations (Hansen 2013 + its critique Tropek 2014, Gorelick/GEE, Gitelson/VARI,
   McGarigal/FRAGSTATS, IPCC carbon factors, Raj/Bengaluru vegetation, Nagendra/Bangalore
   fragmentation, Tahir/CA-Markov, 3x CA-ANN precedent papers [Jain-Delhi,
   Rahman-Dhaka, Baig-Selangor], Torres/Amazon-FCN, ThanhNoi/RF-comparison,
   Boulila/CNN-LSTM), both architecture diagrams (system pipeline + literal MLP
   node diagram), full honest Methodology/Results/Discussion.

Two architecture diagrams exist as PNGs (were built with matplotlib): Fig 1 = the
two-branch CA-ANN system pipeline, Fig 2 = the literal 4→16→8→1 MLP node diagram.

Key correction already made: the model is NOT a CNN — it's a small MLP (CA-ANN) on
4 engineered features, not raw pixels. Keep this distinction precise in any new content.

=== CURRENT ACTIVE TASK (this is what to continue) ===

Faculty said Review 2 was good but missing "one big thing" for a standout final
project. Two suggestions given: (1) "LLM for transfer learning" — corrected: this
should be a pretrained GEOSPATIAL VISION foundation model, not an LLM (LLMs are
text-only, no relevant transfer path to pixel classification). (2) Switch from JPEG/PNG
images to real geospatial data — CORRECT and already root-caused: satellite_fetch.py
currently uses ee.Image.getThumbURL(), a VISUALIZATION endpoint (clips to 0-3000, gamma
1.4, flattens to 8-bit PNG) — meaning FVI has been computed on cosmetically-stretched
display values, not real reflectance, this whole time.

DECISIONS ALREADY LOCKED IN (do not re-litigate unless something breaks):
- Model chosen: Clay v1.5 (~70M params, open weights, OpenRAIL-M license, NATIVE
  Sentinel-2 support — not Prithvi, which routes through HLS/Landsat-harmonization).
- Approach: frozen EMBEDDING EXTRACTION only, NOT fine-tuning. No GPU needed — this
  runs fine on the user's 16GB i5 laptop (~70M params ≈ 280MB, inference-only, ~320
  total patches across the current dataset, cache embeddings to .npy so this only
  runs once ever, not per training run).
- The existing CA-ANN's MLP architecture (16→8) does NOT change — it just receives
  a richer feature vector (current 4 features + Clay's embedding) instead of only 4.
- Known honest tradeoff to state explicitly if asked: the current 4 features are
  purely local/position-agnostic (this was the basis of the "trained on 2 cities works
  anywhere" argument from Review 2 Q&A prep). Clay's embeddings DO encode real lat/lon
  + date. Adding Clay is a genuine trade — richer features, at the cost of that clean
  location-agnostic argument. Say this openly, don't let a judge discover it.

FILES ALREADY WRITTEN AND DELIVERED (exist in outputs, not yet tested by the user
against real GEE/Clay — user has NOT run these yet):
- raster_io.py — shared GeoTIFF read/write utility. Returns REAL reflectance
  (float32, scaled by /10000 per Sentinel-2 SR convention), not 0-255 uint8. Also
  gives real per-pixel lat/lon via pixel_latlon() (impossible with the old PNGs,
  which had no georeferencing at all).
- satellite_fetch_geotiff.py — rewritten fetch script using
  getDownloadURL(format='GEO_TIFF') instead of getThumbURL(). Parallel/new file,
  NOT yet replacing the original satellite_fetch.py — user should test on 1 city/year
  first before fully switching over.
- clay_features.py — frozen Clay embedding extraction. Explicitly marked with
  "VERIFY" comments at the exact lines where Clay's actual current API (import path,
  exact input dict keys, encoder call signature) needs checking against Clay's live
  GitHub repo/example notebook before running, since that repo's API isn't stable
  across releases and I could not test-execute it (sandbox has no network access to
  huggingface.co or earthengine.googleapis.com — all three files are syntax-checked
  only, not run against real GEE/Clay).

IMPORTANT CASCADING SCOPE: switching to GeoTIFF touches EVERY file that currently does
cv2.imread() on the PNGs — main.py (load_satellite/load_nir), test_fvi.py,
train_ca_ann.py (load_all_masks). raster_io.py exists specifically to fix this in one
shared place rather than four. This migration is NOT done yet.

NEXT IMMEDIATE STEPS (in order):
1. User tests satellite_fetch_geotiff.py on one city/year, confirms real .tif values
   land around 0-1 (not 0-255).
2. User confirms Clay checkpoint loads + one patch runs through it successfully,
   fixing the VERIFY-marked lines against Clay's current actual API if needed.
3. Write multiscale_features_v2() — concatenates the existing 4 features with the
   relevant Clay tile embedding per pixel (Clay is patch-level at 224x224, coarser
   than per-pixel — broadcasting each tile's embedding to its pixels is the planned
   approach, not yet built).
4. Migrate main.py / test_fvi.py / train_ca_ann.py call sites to raster_io.py instead
   of cv2.imread, and update FVI/Otsu threshold constants for the new 0-1 reflectance
   scale (they currently assume 0-255 gamma-corrected values).
5. Retrain via train_ca_ann.py, re-run walk-forward validation, report the real
   result honestly either way — same standard as every other experiment in this
   project (the reverted trend-feature experiment, the independent-threshold-design
   replacement, etc.) — a negative result here is a fine, reportable outcome.

=== HOW TO WORK WITH THIS USER ===

Moves fast, high-intensity compressed sessions (built Review 1 AND Review 2 codebases
each in single overnight sessions). Wants direct, concrete, runnable code — not lengthy
hedging. Still values real honesty over reassurance: flag real risks/limitations
plainly (as done throughout: the CA-ANN parity finding, the Hansen granularity
explanation, the location-agnostic tradeoff above), but don't let honesty slow down
forward momentum — give the caveat in one line and keep moving. Always test small
before a full run (one city/year, one patch) rather than committing to a full loop
untested — this pattern has caught real bugs every time so far.