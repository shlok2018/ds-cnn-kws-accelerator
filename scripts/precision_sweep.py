#!/usr/bin/env python3
"""
Precision co-design: int8 vs int4 at the chosen design point (8x8 / glb2048).

This is the HARDWARE half of the precision-accuracy tradeoff -- energy and area
from Timeloop/Accelergy as the operand width drops from 8 to 4 bits (narrower
multiplier + narrower operand storage; the accumulator stays wide).

The ACCURACY half needs the trained model: quantize the DS-CNN to int4, evaluate
top-1 on Google Speech Commands, and drop the number into ACCURACY below to
complete the Pareto. The int8 anchor (~94.4%) is the MLPerf Tiny DS-CNN
reference; int4 is left None on purpose -- do not guess it, measure it.
"""
import os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sweep as S

ROOT = S.ROOT
WEIGHTS = {"conv_standard": 1, "conv_depthwise": 4, "conv_pointwise": 4, "conv_fc": 1}
PRECISIONS = [8, 4]
# Measured top-1 on the MLPerf Tiny KWS test set (4890 samples), post-training
# per-channel WEIGHT-ONLY quantization of the reference model (float32 = 92.17%).
# See kws_eval/eval_precision.py. (Full int4 on activations, or QAT, would shift
# these -- weight-only PTQ is the conservative-but-honest first-order number.)
ACCURACY = {8: 92.29, 4: 87.79}


def per_inference(precision):
    """Weighted per-inference energy (uJ) and area (um2) at 8x8/glb2048."""
    E, A = 0.0, None
    for layer, wt in WEIGHTS.items():
        r = S.run_one(8, 8, 2048, layer, precision=precision, dataflow="ws")
        assert r["ok"], f"{layer} int{precision} failed: {r.get('error')}"
        E += wt * r["energy_uJ"]
        A = r["area_um2"]          # area is layer-independent (fixed arch)
    return E, A


results = {}
for p in PRECISIONS:
    E, A = per_inference(p)
    results[p] = (E, A)
    print(f"int{p}: energy/inf = {E:6.2f} uJ   area = {A:8.0f} um2   "
          f"top-1 = {ACCURACY[p]}")

E8, A8 = results[8]
E4, A4 = results[4]
print(f"\nint8 -> int4:  energy -{100*(E8-E4)/E8:.0f}%   area -{100*(A8-A4)/A8:.0f}%")
print("Pareto is INCOMPLETE until int4 accuracy is measured: energy/area say int4 "
      "is a big win, but KWS accuracy is the price -- fill ACCURACY[4] and re-run.")

# --- figure: energy + area, int8 vs int4 ------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))
labels = ["int8", "int4"]
for ax, vals, ylab, title in [
    (ax1, [E8, E4], "energy per inference (uJ, 45nm)", "Energy"),
    (ax2, [A8/1e6, A4/1e6], "area (mm^2, 45nm)", "Area"),
]:
    bars = ax.bar(labels, vals, color=["tab:blue", "tab:orange"], width=0.6)
    for b, v in zip(bars, vals):
        ax.text(b.get_x()+b.get_width()/2, v, f"{v:.2f}", ha="center", va="bottom")
    ax.set_ylabel(ylab); ax.set_title(title); ax.set_ylim(0, max(vals)*1.18)
red_e = 100*(E8-E4)/E8
red_a = 100*(A8-A4)/A8
fig.suptitle(f"Precision at 8x8/glb2048: int4 cuts energy {red_e:.0f}% & area "
             f"{red_a:.0f}% -- accuracy cost TBD (measure on your model)")
plt.tight_layout()
out = os.path.join(ROOT, "fig_precision.png")
plt.savefig(out, dpi=140)
print(f"wrote {os.path.relpath(out, ROOT)}")
