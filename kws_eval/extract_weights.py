#!/usr/bin/env python3
"""Dump the MLPerf Tiny DS-CNN reference model's weights to a NumPy .npz so the
accelerator inference (sim/run_dscnn_on_accel.py) can run without TensorFlow.
Also prints the layer structure so we can map it correctly."""
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
import numpy as np
from tensorflow import keras

m = keras.models.load_model("trained_models/kws_ref_model")
out = {}
print("=== layer structure ===")
for i, l in enumerate(m.layers):
    ws = l.get_weights()
    print(f"{i:02d} {l.__class__.__name__:22s} {l.name:20s} "
          f"{[list(w.shape) for w in ws]}")
    for j, w in enumerate(ws):
        out[f"L{i:02d}_{l.__class__.__name__}_{j}"] = w.astype(np.float32)

np.savez_compressed("dscnn_weights.npz", **out)
print(f"\nsaved dscnn_weights.npz with {len(out)} arrays")
