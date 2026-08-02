#!/usr/bin/env python3
"""
Accuracy-energy Pareto for the DS-CNN at int8 vs int4.

Accuracy: measured top-1 on the MLPerf Tiny KWS test set (4890 samples) with
post-training, per-output-channel, WEIGHT-ONLY quantization of the reference
model (see kws_eval/eval_precision.py). Energy: per-inference at the chosen
8x8/glb2048 design point from precision_sweep.py (Accelergy, 45 nm).

Caveats shown on the plot: (1) weights are quantized but activations are left in
float, whereas the accelerator's energy point assumes int4 on both operands, so
the true int4-both accuracy would be lower without (2) quantization-aware
training, which typically recovers most of the int4 drop.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

ACC   = {"int8": 92.29, "int4": 87.79}   # measured top-1 (%), weight-only PTQ
FLOAT = 92.17                            # float32 reference top-1 (%)
ENERGY = {"int8": 17.18, "int4": 8.65}   # uJ/inference @ 8x8/glb2048 (Accelergy 45nm)

xs = [ENERGY["int8"], ENERGY["int4"]]
ys = [ACC["int8"], ACC["int4"]]

fig, ax = plt.subplots(figsize=(6.6, 4.7))
ax.plot(xs, ys, "-o", color="tab:purple", markersize=9)
for k in ACC:
    ax.annotate(f"{k}\n{ACC[k]:.1f}%, {ENERGY[k]:.1f} uJ",
                (ENERGY[k], ACC[k]), textcoords="offset points",
                xytext=(10, -6 if k == "int8" else 6), fontsize=9)
ax.axhline(FLOAT, ls="--", color="gray", lw=1, label=f"float32 ceiling ({FLOAT:.1f}%)")
ax.set_xlabel("energy per inference ($\\mu$J, 45 nm)")
ax.set_ylabel("top-1 accuracy (%)")
ax.set_title("Accuracy vs energy: int4 halves energy for ~4.5 pts of accuracy\n"
             "(weight-only PTQ; QAT would recover much of the int4 drop)")
ax.set_ylim(85, 94)
ax.legend(loc="lower right")
ax.grid(alpha=0.3)
plt.tight_layout()
out = os.path.join(ROOT, "fig_accuracy_energy.png")
plt.savefig(out, dpi=140)
print("measured top-1: float32 %.2f, int8 %.2f, int4 %.2f" % (FLOAT, ACC["int8"], ACC["int4"]))
print("wrote", os.path.relpath(out, ROOT))
