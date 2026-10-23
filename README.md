# ForestWatch India

A CA-ANN framework for urban forest cover monitoring and multi-year prediction using Sentinel-2 imagery.

Final-year B.Tech CSE thesis project. Yuvraj Singh Hajari (23BCE1995), VIT Chennai. Guide: Kumar K.

## What it does

ForestWatch India detects and forecasts urban forest cover change across four Indian cities (Chennai, Bengaluru, Delhi, Kolkata) from Sentinel-2 surface reflectance. It combines:

- **FVI (Forest Vegetation Index)** — a single-formula index, `(NIR+G−R−B)/(NIR+G+R+B)`, that separates water, bare land and vegetation in one pass via three-class Otsu thresholding, in place of the two-index (greenness + water mask) design most published pipelines use. Benchmarked directly against the literature-standard NDVI+NDWI combination: a measurable, city-dependent advantage in water-heavy scenes (0.00% vs. 3.37% of water misclassified as forest in coastal Chennai), and an open, honestly-reported tradeoff in land-detection coverage elsewhere. Both results are in the IEEE paper.
- **CA-ANN prediction** — a hybrid cellular automata / artificial neural network model that separates *how much* land cover is expected to change (a corrected historical Markov transition rate) from *which pixels* change (a trained MLP classifier), validated through walk-forward testing against real held-out years.
- **Clay v1.5 embeddings** — a frozen geospatial foundation model, integrated as an additional feature source and evaluated rigorously across all four cities; the evaluation, including a negative result in three of the four cities and the eight-seed ensemble that stabilizes Bengaluru's result, is documented in the IEEE paper below.

The system is served as a FastAPI backend with a React frontend, and also provides a conservation-priority ranking (named against real local landmarks, not bare coordinates), a continuous risk heatmap, and external validation against Hansen Global Forest Change.

The frontend itself is a guided walkthrough rather than a bare dashboard: a hand-traced India map, a per-city intro that fetches and narrates ten years of real imagery one year at a time, a step that teaches the FVI detection method with live numbers before any model is involved, scroll-driven diagrams explaining the CA-ANN pipeline stage by stage, and a closing page built specifically to compare this work's validated Kappa against six published CA-ANN studies.

## Architecture

```
main.py                      FastAPI backend: 8 analysis endpoints, CA-ANN inference, in-memory response cache
raster_io.py                 GeoTIFF reflectance loading
clay_features.py             Clay v1.5 feature extraction
extract_clay_embeddings.py   Offline script: tiles imagery, extracts + PCA-reduces Clay embeddings
extract_clay_patch_tokens.py Diagnostic: finer-grained per-patch (not per-tile) Clay embeddings
train_ca_ann.py              Walk-forward training and validation for the pooled CA-ANN
bengaluru_ensemble_experiment.py  8-seed ensemble training for Bengaluru's specialized model
test_fvi_vs_ndvi.py          FVI vs. the literature-standard NDVI+NDWI, run directly on fetched data
satellite_fetch*.py          Sentinel-2 / Hansen GFC fetch scripts (Google Earth Engine)
forestwatch-frontend/        React + Vite frontend
  src/components/Landing.jsx       Traced India map, click-to-zoom city picker
  src/components/intro/            Per-city guided fetch + FVI teaching steps
  src/components/sections/         Detection, Compare, Predict, More Features, Big Question
  src/components/diagrams/         Pre-rendered CA-ANN pipeline diagrams (not generated at runtime)
  public/diagrams/                 The diagram PNGs themselves
```

Trained model weights (`ca_ann_model*.joblib`), validation stats (`ca_ann_model_stats*.json`), the Bengaluru ensemble, and the PCA basis are committed directly since they're cheap to store and expensive to regenerate; fetched imagery (`static/`, `*.tif`, `*.png`) and the Clay checkpoint (`checkpoints/`) are gitignored and regenerated via the fetch/extraction scripts. The frontend's own diagram and image assets under `forestwatch-frontend/public/` and `src/` are committed despite the blanket `*.png` rule above, since they're hand-built site content, not regenerated output.

## Setup

### Backend

```bash
python -m venv venv
venv\Scripts\activate          # Windows; use `source venv/bin/activate` on macOS/Linux
pip install -r requirements.txt

cp .env.example .env           # fill in GEE_PROJECT
earthengine authenticate       # one-time Earth Engine login

python main.py                 # serves on http://127.0.0.1:8000
```

### Frontend

```bash
cd forestwatch-frontend
npm install
npm run dev                    # serves on http://localhost:5173
```

The frontend reads the backend URL from `VITE_API_URL`, defaulting to `http://127.0.0.1:8000` if unset.

### Fetching new imagery / retraining

Imagery and Hansen reference data are not committed. To populate `static/` and retrain from scratch:

```bash
python satellite_fetch_geotiff.py   # analysis-ready Sentinel-2 reflectance per city-year
python satellite_fetch.py           # display-only PNG composites
python hansen_fetch.py              # Hansen Global Forest Change reference
python extract_clay_embeddings.py   # Clay v1.5 embeddings + PCA (requires checkpoints/, ~5GB)
python train_ca_ann.py              # trains the pooled CA-ANN and writes validation stats
```

## API

| Endpoint | Purpose |
|---|---|
| `GET /cities` | List configured cities |
| `GET /years/{city}` | Available years for a city |
| `GET /analyze/{city}/{year}` | Single-year forest detection |
| `GET /compare/{city}/{year_a}/{year_b}` | Two-year gain/loss comparison |
| `GET /predict/{city}/{target_year}` | CA-ANN forward projection, with validation stats |
| `GET /risk/{city}` | Continuous change-risk heatmap |
| `GET /conservation-priority/{city}` | Area- and risk-ranked patch list |
| `GET /validate-hansen/{city}/{year}` | External validation against Hansen GFC |

## Documentation

- `ForestWatch_India_IEEE_Paper.docx` — IEEE double-column paper covering methodology, the Clay ablation, the FVI-vs-NDVI+NDWI validation, and comparison against six published CA-ANN and cellular automata studies.
- `ForestWatch_India_Review_Deck.pptx` — review presentation deck: system architecture, the full prediction workflow, every CA-ANN pipeline diagram individually, and results.

The full VIT-format thesis report submitted for this project's academic evaluation is intentionally not included in this public repository.

## Known limitations

Clay's embeddings improve prediction in only one of the four validated cities (Bengaluru); the other three show a measured negative delta. Long-horizon (6-year) accuracy trails published cellular automata studies that model monotonic urban growth, though extending the comparison to two published studies that do model a bidirectional forest class shows bidirectionality alone does not explain the gap; calibration history length is the more likely factor, detailed in the IEEE paper. FVI's water-detection advantage over the standard NDVI+NDWI approach is real but city-dependent, and in the cities where it does not help with water, FVI also detects noticeably less total forest on land than NDVI+NDWI, a tradeoff that is reported rather than hidden. All of these results are discussed in detail in the IEEE paper rather than omitted.
