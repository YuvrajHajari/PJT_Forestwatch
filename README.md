# ForestWatch India

A CA-ANN framework for urban forest cover monitoring and multi-year prediction using Sentinel-2 imagery.

Final-year B.Tech CSE thesis project. Yuvraj Singh Hajari (23BCE1995), VIT Chennai. Guide: Kumar K.

## What it does

ForestWatch India detects and forecasts urban forest cover change across four Indian cities (Chennai, Bengaluru, Delhi, Kolkata) from Sentinel-2 surface reflectance. It combines:

- **FVI (Forest Vegetation Index)** — a single-formula index, `(NIR+G−R−B)/(NIR+G+R+B)`, that separates water, bare land and vegetation in one pass via three-class Otsu thresholding.
- **CA-ANN prediction** — a hybrid cellular automata / artificial neural network model that separates *how much* land cover is expected to change (a corrected historical Markov transition rate) from *which pixels* change (a trained MLP classifier), validated through walk-forward testing against real held-out years.
- **Clay v1.5 embeddings** — a frozen geospatial foundation model, integrated as an additional feature source and evaluated rigorously across all four cities; the evaluation, including a negative result in three of the four cities and the eight-seed ensemble that stabilizes Bengaluru's result, is documented in the IEEE paper below.

The system is served as a FastAPI backend with a React frontend, and also provides a conservation-priority ranking, a continuous risk heatmap, and external validation against Hansen Global Forest Change.

## Architecture

```
main.py                      FastAPI backend: 8 analysis endpoints, CA-ANN inference
raster_io.py                 GeoTIFF reflectance loading
clay_features.py             Clay v1.5 feature extraction
extract_clay_embeddings.py   Offline script: tiles imagery, extracts + PCA-reduces Clay embeddings
train_ca_ann.py              Walk-forward training and validation for the pooled CA-ANN
bengaluru_ensemble_experiment.py  8-seed ensemble training for Bengaluru's specialized model
satellite_fetch*.py          Sentinel-2 / Hansen GFC fetch scripts (Google Earth Engine)
forestwatch-frontend/        React + Vite frontend
```

Trained model weights (`ca_ann_model*.joblib`), validation stats (`ca_ann_model_stats*.json`), the Bengaluru ensemble, and the PCA basis are committed directly since they're cheap to store and expensive to regenerate; fetched imagery (`static/`, `*.tif`, `*.png`) and the Clay checkpoint (`checkpoints/`) are gitignored and regenerated via the fetch/extraction scripts.

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

- `ForestWatch_India_IEEE_Paper.docx` — IEEE double-column paper covering methodology, the Clay ablation, and comparison against three published CA-ANN studies.
- `ForestWatch_India_Project_Report.docx` — full VIT-format thesis report.
- `ForestWatch_India_Review_Deck.pptx` — review presentation deck.

## Known limitations

Clay's embeddings improve prediction in only one of the four validated cities (Bengaluru); the other three show a measured negative delta. Long-horizon (6-year) accuracy trails published cellular automata studies that model monotonic urban growth rather than bidirectional forest change. Both results are reported and discussed in detail in the IEEE paper rather than omitted.
