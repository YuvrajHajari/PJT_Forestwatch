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
    # Download the v1.5 checkpoint (see Clay's README for the current
    # Hugging Face path -- this moves between releases, verify before
    # running):
    #   https://huggingface.co/made-with-clay/Clay

VERIFY BEFORE RUNNING -- Clay's exact module/class names and expected
input dict keys are NOT guaranteed stable across their releases (this is
an active research repo, not a versioned stable library). The import
below and the `datacube` dict shape reflect Clay's documented pattern as
of this writing; if the import fails, check the "Getting Started" /
embeddings example notebook in the Clay repo for the current exact
signature and adjust load_clay() and build_datacube() accordingly --
the rest of this file (patch extraction, feature assembly, MLP handoff)
does not depend on those specifics and needs no changes.
"""
import os
import numpy as np
import torch

# --- VERIFY: exact import path per Clay's current repo/notebook ---
from claymodel.module import ClayMAEModule  # noqa: E402

CLAY_CHECKPOINT = os.environ.get("CLAY_CHECKPOINT_PATH", "checkpoints/clay-v1.5.ckpt")
PATCH_SIZE = 224  # Clay's default tile size -- verify against the checkpoint's config
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_model_cache = {"model": None}


def load_clay():
    """Loads the frozen Clay encoder once, caches it for reuse across
    every patch/every city/every year in a single process."""
    if _model_cache["model"] is None:
        model = ClayMAEModule.load_from_checkpoint(
            CLAY_CHECKPOINT, metadata_path="configs/metadata.yaml"
        )
        model.eval()
        model.to(DEVICE)
        for p in model.parameters():
            p.requires_grad = False  # frozen -- inference only, confirms no accidental fine-tuning
        _model_cache["model"] = model
    return _model_cache["model"]


def build_datacube(rgb_nir_patch: np.ndarray, lat: float, lon: float, date):
    """
    rgb_nir_patch: (4, PATCH_SIZE, PATCH_SIZE) float32 reflectance
    (R, G, B, NIR -- from raster_io.load_satellite / load_nir, real
    reflectance units, NOT the old 0-255 scale).

    Returns the dict shape Clay's encoder expects: pixel values, plus
    real location/time metadata (Clay uses these as embeddings inputs,
    unlike your current 4 features, which carry no spatial/temporal
    context at all -- see the "why this matters" note at the bottom).

    VERIFY: exact key names ("pixels", "time", "latlon", platform string)
    against Clay's current datamodule/example notebook before running.
    """
    return {
        "pixels": torch.from_numpy(rgb_nir_patch).float().unsqueeze(0).to(DEVICE),
        "time": torch.tensor([[date.year, date.month, date.day]]).to(DEVICE),
        "latlon": torch.tensor([[lat, lon]]).float().to(DEVICE),
        "platform": "sentinel-2-l2a",
    }


def extract_patch_embedding(rgb_nir_patch: np.ndarray, lat: float, lon: float, date) -> np.ndarray:
    """
    Runs one patch through the frozen encoder, mean-pools the output
    patch tokens into a single fixed-length vector.

    Returns: 1D float32 array (length depends on the checkpoint -- Clay
    v1.5 base is a few hundred dimensions; print embedding.shape on your
    first successful run and record it, since downstream code needs
    that exact length).
    """
    model = load_clay()
    datacube = build_datacube(rgb_nir_patch, lat, lon, date)
    with torch.no_grad():
        # VERIFY: exact call -- Clay's docs show patterns like
        # model.model.encoder(datacube) returning patch tokens; confirm
        # against the current embeddings example notebook.
        embeddings = model.model.encoder(datacube)
        pooled = embeddings.mean(dim=1)  # mean-pool patch tokens -> one vector per input patch
    return pooled.squeeze(0).cpu().numpy()


def extract_features_for_mask_grid(rgb: np.ndarray, nir: np.ndarray, meta: dict,
                                     date, stride: int = 224) -> dict:
    """
    Tiles a full 1024x1024 city image into PATCH_SIZE chunks, extracts
    one Clay embedding per tile, and returns {(row, col): embedding}.

    This gives COARSER spatial resolution than your current per-pixel
    features (one vector per 224x224 tile, not per pixel) -- that's an
    honest tradeoff of this approach, not a bug: Clay is a patch-level
    model. The integration step (next file) is where you decide how to
    combine this coarse, learned signal with your existing fine-grained
    per-pixel density/edge features, e.g. by broadcasting each tile's
    embedding to every pixel within it as extra constant-valued columns.
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
            out[(row, col)] = extract_patch_embedding(patch, lat, lon, date)
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