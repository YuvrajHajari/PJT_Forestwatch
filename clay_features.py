"""
clay_features.py -- Extracts frozen Clay v1.5 embeddings per pixel
neighborhood, as a drop-in replacement/supplement for multiscale_features()
in main.py / train_ca_ann.py.

APPROACH: "Pattern B" (embeddings), not fine-tuning.
  - Clay's encoder runs ONCE per patch, weights frozen, CPU is fine.
  - No GPU required, no training loop for Clay itself.
  - Output: a fixed-length vector per patch, which becomes additional
    input features for your EXISTING small MLPClassifier -- that
    network's architecture (16, 8 hidden units) does not need to change,
    it just receives a richer feature vector than the current 4 numbers.

SETUP (run once, on your machine -- not in this script):
    pip install git+https://github.com/Clay-foundation/model.git
    # Download the v1.5 checkpoint:
    #   https://huggingface.co/made-with-clay/Clay/blob/main/clay-v1.5.ckpt
    # And Clay's metadata.yaml (band mean/std/wavelength per platform,
    # needed to build the model's per-platform architecture):
    #   https://github.com/Clay-foundation/model/blob/main/configs/metadata.yaml
    # Set CLAY_CHECKPOINT_PATH / CLAY_METADATA_PATH env vars, or drop
    # them at the default paths below.

VERIFIED against Clay's actual source (2026-09) -- not guessed:
claymodel/module.py, claymodel/datamodule.py, configs/metadata.yaml, and
docs/tutorials/inference.ipynb (their real single-scene embedding-extraction
example -- the one closest to our use case) on github.com/Clay-foundation/model
main branch, plus stacchip/processors/prechip.py (normalize_timestamp /
normalize_latlon, reimplemented inline below so this file doesn't need the
full stacchip package and its STAC/lancedb dependency tree just for two
sin/cos formulas).

CRITICAL: pixel scale. Clay's own band stats (mean/std below) are in raw
~0-10000 Sentinel-2 SR digital-number scale -- the SAME scale raster_io.py
deliberately divides away (by SR_SCALE) to get 0-1 reflectance for FVI/Otsu.
Feed Clay the /10000 reflectance floats and you get silent garbage (z-scores
around -600), not an error. This file re-multiplies by SR_SCALE internally
so callers can keep passing raster_io's 0-1 reflectance consistently
everywhere else in the codebase.
"""
import datetime
import math
import os

import numpy as np
import torch
import yaml

from claymodel.module import ClayMAEModule
from raster_io import SR_SCALE

CLAY_CHECKPOINT = os.environ.get("CLAY_CHECKPOINT_PATH", "checkpoints/clay-v1.5.ckpt")
CLAY_METADATA_PATH = os.environ.get("CLAY_METADATA_PATH", "configs/metadata.yaml")
PATCH_SIZE = 256  # Clay's tested chip size (docs/tutorials/*.ipynb), not ViT's default 224
PLATFORM = "sentinel-2-l2a"
BAND_NAMES = ["red", "green", "blue", "nir"]  # matches raster_io's R,G,B + NIR order
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_model_cache = {"model": None}
_metadata_cache = {"meta": None}


def load_clay():
    """Loads the frozen Clay v1.5 (large) encoder once, caches it for reuse
    across every patch/every city/every year in a single process.

    model_size/dolls/doll_weights/mask_ratio/shuffle must match how v1.5
    was trained (see Clay's docs/tutorials/embeddings.ipynb, the notebook
    that actually loads clay-v1.5.ckpt) -- defaults on ClayMAEModule are
    model_size="base", mask_ratio=0.75, both wrong for this checkpoint."""
    if _model_cache["model"] is None:
        model = ClayMAEModule.load_from_checkpoint(
            checkpoint_path=CLAY_CHECKPOINT,
            model_size="large",
            metadata_path=CLAY_METADATA_PATH,
            dolls=[16, 32, 64, 128, 256, 768, 1024],
            doll_weights=[1, 1, 1, 1, 1, 1, 1],
            mask_ratio=0.0,
            shuffle=False,
        )
        model.eval()
        model.to(DEVICE)
        for p in model.parameters():
            p.requires_grad = False  # frozen -- inference only, confirms no accidental fine-tuning
        _model_cache["model"] = model
    return _model_cache["model"]


def _load_band_metadata():
    """Pulls mean/std/wavelength for our 4 bands out of Clay's own
    metadata.yaml, instead of hardcoding duplicate numbers that could
    drift out of sync with whatever checkpoint/metadata pair is in use."""
    if _metadata_cache["meta"] is None:
        with open(CLAY_METADATA_PATH) as f:
            full_meta = yaml.safe_load(f)
        bands = full_meta[PLATFORM]["bands"]
        _metadata_cache["meta"] = {
            "mean": np.array([bands["mean"][b] for b in BAND_NAMES], dtype=np.float32),
            "std": np.array([bands["std"][b] for b in BAND_NAMES], dtype=np.float32),
            "waves": np.array([bands["wavelength"][b] for b in BAND_NAMES], dtype=np.float32),
        }
    return _metadata_cache["meta"]


def normalize_timestamp(date: datetime.datetime):
    """Sin/cos encoding of (ISO week, hour-of-day). Reimplemented from
    stacchip.processors.prechip.normalize_timestamp -- same formula, no
    added dependency."""
    week = date.isocalendar().week * 2 * math.pi / 52
    hour = date.hour * 2 * math.pi / 24
    return (math.sin(week), math.cos(week)), (math.sin(hour), math.cos(hour))


def normalize_latlon(lat: float, lon: float):
    """Sin/cos encoding of lat/lon. Reimplemented from
    stacchip.processors.prechip.normalize_latlon, same formula."""
    lat_rad = lat * math.pi / 180
    lon_rad = lon * math.pi / 180
    return (math.sin(lat_rad), math.cos(lat_rad)), (math.sin(lon_rad), math.cos(lon_rad))


def build_datacube(rgb_nir_patch_reflectance: np.ndarray, lat: float, lon: float,
                    date: datetime.datetime, gsd: float):
    """
    rgb_nir_patch_reflectance: (4, PATCH_SIZE, PATCH_SIZE) float32, in the
    SAME 0-1 reflectance units raster_io.py produces everywhere else
    (R, G, B, NIR order). Rescaled back to raw ~0-10000 DN internally,
    then z-score normalized with Clay's own per-band mean/std -- both
    steps Clay's encoder actually expects (see module docstring).

    gsd: real per-pixel ground resolution in meters for THIS fetch (e.g.
    ~11.72 for a 6km-radius/1024px city image), NOT metadata.yaml's
    generic Sentinel-2 gsd=10 -- our images are resampled to a custom
    grid, so the nominal sensor gsd is the wrong number here.

    Returns the dict shape Clay's encoder actually expects: pixels, time,
    latlon, gsd, waves (verified against docs/tutorials/inference.ipynb's
    prep_datacube -- "platform" is NOT a datacube key, it's only used to
    look up band stats before building this dict).
    """
    meta = _load_band_metadata()
    dn_patch = rgb_nir_patch_reflectance * SR_SCALE  # reflectance -> raw DN scale Clay's stats assume

    mean = meta["mean"].reshape(-1, 1, 1)
    std = meta["std"].reshape(-1, 1, 1)
    normalized = (dn_patch - mean) / std

    week_norm, hour_norm = normalize_timestamp(date)
    lat_norm, lon_norm = normalize_latlon(lat, lon)

    pixels = torch.from_numpy(normalized.astype(np.float32)).unsqueeze(0).to(DEVICE)
    time = torch.tensor(np.hstack([week_norm, hour_norm]), dtype=torch.float32,
                         device=DEVICE).unsqueeze(0)
    latlon = torch.tensor(np.hstack([lat_norm, lon_norm]), dtype=torch.float32,
                           device=DEVICE).unsqueeze(0)
    waves = torch.tensor(meta["waves"], dtype=torch.float32, device=DEVICE)
    gsd_t = torch.tensor(gsd, dtype=torch.float32, device=DEVICE)

    return {"pixels": pixels, "time": time, "latlon": latlon, "gsd": gsd_t, "waves": waves}


def extract_patch_embedding(rgb_nir_patch_reflectance: np.ndarray, lat: float, lon: float,
                             date: datetime.datetime, gsd: float) -> np.ndarray:
    """
    Runs one patch through the frozen encoder, returns the CLS token as
    the single fixed-length embedding for the whole patch (this is
    Clay's own recommended single-vector-per-chip approach -- see
    generate_embeddings() in docs/tutorials/inference.ipynb: "The first
    embedding is the class token, which is the overall single
    embedding" -- NOT a mean-pool over all patch tokens, which is a
    different use case for a per-pixel spatial embedding map).

    Returns: 1D float32 array. Print embedding.shape on your first
    successful run -- Clay v1.5 large's embedding dim needs recording
    for downstream feature-vector-length code.
    """
    model = load_clay()
    datacube = build_datacube(rgb_nir_patch_reflectance, lat, lon, date, gsd)
    with torch.no_grad():
        unmsk_patch, unmsk_idx, msk_idx, msk_matrix = model.model.encoder(datacube)
        cls_embedding = unmsk_patch[:, 0, :]
    return cls_embedding.squeeze(0).cpu().numpy()


def extract_features_for_mask_grid(rgb: np.ndarray, nir: np.ndarray, meta: dict,
                                     date: datetime.datetime, gsd: float,
                                     stride: int = PATCH_SIZE) -> dict:
    """
    Tiles a full 1024x1024 city image into PATCH_SIZE chunks, extracts
    one Clay embedding per tile, and returns {(row, col): embedding}.

    This gives COARSER spatial resolution than your current per-pixel
    features (one vector per 256x256 tile, not per pixel) -- that's an
    honest tradeoff of this approach, not a bug: Clay is a patch-level
    model. The integration step (next file) is where you decide how to
    combine this coarse, learned signal with your existing fine-grained
    per-pixel density/edge features, e.g. by broadcasting each tile's
    embedding to every pixel within it as extra constant-valued columns.

    rgb, nir: reflectance arrays as returned by raster_io.load_satellite /
    load_nir (0-1 units) -- scale conversion happens inside build_datacube.
    """
    from raster_io import pixel_latlon  # local import to avoid a hard dependency for callers who only need Clay itself

    h, w = rgb.shape[0], rgb.shape[1]
    out = {}
    for row in range(0, h - PATCH_SIZE + 1, stride):
        for col in range(0, w - PATCH_SIZE + 1, stride):
            rgb_patch = rgb[row:row+PATCH_SIZE, col:col+PATCH_SIZE, :]
            nir_patch = nir[row:row+PATCH_SIZE, col:col+PATCH_SIZE]
            patch = np.concatenate(
                [np.transpose(rgb_patch, (2, 0, 1)), nir_patch[np.newaxis, :, :]], axis=0
            )  # -> (4, PATCH_SIZE, PATCH_SIZE)
            lat, lon = pixel_latlon(meta, row + PATCH_SIZE // 2, col + PATCH_SIZE // 2)
            out[(row, col)] = extract_patch_embedding(patch, lat, lon, date, gsd)
    return out


# ---------------------------------------------------------------------
# WHY location/time inputs matter here, worth saying out loud in review:
# your current 4 hand-engineered features (density x3 scales + edge
# distance) are purely LOCAL and carry no absolute position or season --
# that's exactly what made the "trained on 2 cities, works anywhere"
# argument valid. Clay's embeddings, by contrast, DO encode real lat/lon
# and date. That's a genuine trade you're making by adding this: richer,
# more expressive features, at the cost of the clean "fully
# location-agnostic" argument from your last review. Worth stating this
# explicitly rather than letting a judge discover the tension themselves.
# ---------------------------------------------------------------------
