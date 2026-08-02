#!/usr/bin/env python3
"""
Measure DS-CNN keyword-spotting accuracy at float32 / int8 / int4.

Uses the MLPerf Tiny reference model and its own data pipeline (tfds
speech_commands, same MFCC preprocessing), so the test set matches the benchmark.
Quantization here is post-training, symmetric, per-output-channel, WEIGHTS ONLY
(activations left in float). That is the honest, reproducible first-order number;
the accelerator also quantizes activations, and aggressive int4 would normally
use quantization-aware training to recover accuracy -- both noted as caveats.
"""
import os, sys, numpy as np
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
# TF 2.16+ defaults to Keras 3, which cannot load the reference legacy SavedModel.
# Route tensorflow.keras to the Keras-2 compat package (pip install tf_keras).
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
import tensorflow as tf
from tensorflow import keras

sys.argv = [sys.argv[0]]                 # let parse_command() use its defaults
import kws_util
import get_dataset as kws_data

Flags, _ = kws_util.parse_command()
Flags.model_init_path = os.path.abspath("trained_models/kws_ref_model")
DATA = os.environ.get("KWS_DATA", os.path.abspath("data"))
os.environ["TFDS_DATA_DIR"] = DATA
Flags.data_dir = DATA
print(f"[cfg] data_dir={DATA}", flush=True)

CACHE = "test_data.npz"
if os.path.exists(CACHE):
    d = np.load(CACHE); X, y = d["X"], d["y"]
    print(f"[data] loaded cached test set {X.shape}", flush=True)
else:
    print("[data] downloading + preprocessing speech_commands test split ...", flush=True)
    _, ds_test, _ = kws_data.get_training_data(Flags)
    Xs, ys = [], []
    for xb, yb in ds_test:
        Xs.append(xb.numpy()); ys.append(yb.numpy())
    X = np.concatenate(Xs); y = np.concatenate(ys)
    np.savez_compressed(CACHE, X=X, y=y)
    print(f"[data] cached test set {X.shape}", flush=True)

model = keras.models.load_model(Flags.model_init_path)

def accuracy(m):
    preds = np.argmax(m.predict(X, batch_size=256, verbose=0), axis=1)
    return float((preds == y).mean())

def quantize_weights(src, bits):
    """Post-training symmetric per-output-channel weight quantization."""
    qm = keras.models.clone_model(src)
    qm.set_weights(src.get_weights())
    qmax = 2 ** (bits - 1) - 1            # 127 (int8), 7 (int4)
    for layer in qm.layers:
        ws = layer.get_weights()
        if not ws:
            continue
        out = []
        for arr in ws:
            if arr.ndim >= 2:            # conv/dense kernels -> quantize
                ax = tuple(range(arr.ndim - 1))
                scale = np.max(np.abs(arr), axis=ax, keepdims=True) / qmax
                scale = np.where(scale == 0, 1e-8, scale)
                q = np.round(arr / scale).clip(-qmax - 1, qmax) * scale
                out.append(q.astype(np.float32))
            else:                        # biases / BN params -> keep full precision
                out.append(arr)
        layer.set_weights(out)
    return qm

print(f"RESULT float32          {accuracy(model)*100:.2f}", flush=True)
for bits in (8, 4):
    print(f"RESULT int{bits}_weights      {accuracy(quantize_weights(model, bits))*100:.2f}",
          flush=True)
