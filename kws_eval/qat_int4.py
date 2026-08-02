#!/usr/bin/env python3
"""
Quantization-aware training (QAT) for int4 WEIGHTS on the DS-CNN KWS model, to
compare against the weight-only post-training int4 number from eval_precision.py.

Same data pipeline and test set as the eval script (so it is apples-to-apples).
Weights only are quantized to 4 bits (activations left in float), matching the
PTQ comparison; a short fine-tune lets the network adapt to the 4-bit grid.
Fine-tuning is deliberately bounded (few epochs, capped steps) because it runs on
an emulated CPU -- it is a best-effort recovery number, not a tuned result.
"""
import os, sys, numpy as np
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
import tensorflow as tf
from tensorflow import keras
import tensorflow_model_optimization as tfmot
from tensorflow_model_optimization.python.core.quantization.keras.quantizers \
    import LastValueQuantizer

sys.argv = [sys.argv[0]]
import kws_util, get_dataset as kws_data
Flags, _ = kws_util.parse_command()
Flags.model_init_path = os.path.abspath("trained_models/kws_ref_model")
DATA = os.environ.get("KWS_DATA", os.path.abspath("data"))
os.environ["TFDS_DATA_DIR"] = DATA
Flags.data_dir = DATA

# --- test set (cached by eval_precision.py) ---------------------------------
d = np.load("test_data.npz"); Xte, yte = d["X"], d["y"]
print(f"[data] test set {Xte.shape}", flush=True)

# --- training data (same pipeline as the benchmark) -------------------------
ds_train, _, ds_val = kws_data.get_training_data(Flags)
print("[data] train/val pipelines ready", flush=True)

model = keras.models.load_model(Flags.model_init_path)

def acc(m):
    return float((np.argmax(m.predict(Xte, batch_size=256, verbose=0), axis=1) == yte).mean())

print(f"RESULT float32              {acc(model)*100:.2f}", flush=True)

# --- weight-only 4-bit QAT config -------------------------------------------
Q = tfmot.quantization.keras
def w4():
    return LastValueQuantizer(num_bits=4, per_axis=True, symmetric=True, narrow_range=True)

class _Base(Q.QuantizeConfig):
    def get_activations_and_quantizers(self, layer): return []
    def set_quantize_activations(self, layer, q): pass
    def get_output_quantizers(self, layer): return []
    def get_config(self): return {}

class KernelW4(_Base):
    def get_weights_and_quantizers(self, layer): return [(layer.kernel, w4())]
    def set_quantize_weights(self, layer, qw): layer.kernel = qw[0]

class DepthwiseW4(_Base):
    def get_weights_and_quantizers(self, layer): return [(layer.depthwise_kernel, w4())]
    def set_quantize_weights(self, layer, qw): layer.depthwise_kernel = qw[0]

def annotate(layer):
    if isinstance(layer, keras.layers.DepthwiseConv2D):
        return Q.quantize_annotate_layer(layer, DepthwiseW4())
    if isinstance(layer, (keras.layers.Conv2D, keras.layers.Dense)):
        return Q.quantize_annotate_layer(layer, KernelW4())
    return layer

annotated = keras.models.clone_model(model, clone_function=annotate)
annotated.set_weights(model.get_weights())          # carry the float weights over
with Q.quantize_scope({"KernelW4": KernelW4, "DepthwiseW4": DepthwiseW4}):
    qat = Q.quantize_apply(annotated)

print(f"RESULT int4_qat_epoch0      {acc(qat)*100:.2f}   (before fine-tune)", flush=True)

qat.compile(optimizer=keras.optimizers.Adam(1e-4),
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
            metrics=["accuracy"])

EPOCHS, STEPS = 5, 300      # bounded for emulated CPU
for e in range(EPOCHS):
    qat.fit(ds_train, epochs=1, steps_per_epoch=STEPS, validation_data=ds_val,
            validation_steps=40, verbose=2)
    print(f"RESULT int4_qat_epoch{e+1}      {acc(qat)*100:.2f}", flush=True)

print("DONE", flush=True)
