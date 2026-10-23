"""One-off probe: confirm the shape of Clay's full patch-token grid (not
just the CLS token) and get a slightly more reliable per-patch timing
average, before committing to a full re-extraction run."""
import time
import datetime
import numpy as np
import torch

import clay_features as cf

t0 = time.time()
model = cf.load_clay()
print(f"model load: {time.time()-t0:.1f}s")

patch = np.random.rand(4, 256, 256).astype(np.float32) * 0.3
datacube = cf.build_datacube(patch, 12.97, 77.59, datetime.datetime(2025, 3, 1), 11.72)

times = []
for i in range(3):
    t0 = time.time()
    with torch.no_grad():
        unmsk_patch, unmsk_idx, msk_idx, msk_matrix = model.model.encoder(datacube)
    times.append(time.time() - t0)
print(f"unmsk_patch shape: {tuple(unmsk_patch.shape)}  (batch, 1+n_patch_tokens, dim)")
print(f"per-call times: {[round(t,2) for t in times]}, mean: {np.mean(times):.2f}s")
