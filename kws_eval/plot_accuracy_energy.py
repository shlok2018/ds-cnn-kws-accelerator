#!/usr/bin/env python3
"""Accuracy-energy Pareto for the DS-CNN KWS accelerator, from measured numbers.

Accuracy: measured on the MLPerf Tiny speech_commands test set (4890 clips).
Energy:   per-inference at the 8x8/glb2048 design point (Accelergy, 45nm), from
          the precision sweep (int4 halves the int8 energy).
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# (label, energy_uJ, accuracy_%, marker-note)
E8, E4 = 17.18, 8.65                     # per-inference energy, int8 vs int4
pts = [
    ("int8 (PTQ)",  E8, 92.29, "tab:blue"),
    ("int4 (PTQ)",  E4, 87.79, "tab:red"),
    ("int4 (QAT)",  E4, 91.62, "tab:green"),
]
FLOAT_ACC = 92.17

fig, ax = plt.subplots(figsize=(6.4, 4.6))
ax.axhline(FLOAT_ACC, ls=":", color="gray", lw=1)
ax.text(E8, FLOAT_ACC + 0.15, f"float32 ref = {FLOAT_ACC:.1f}%", color="gray", ha="right", fontsize=8)

for label, e, a, c in pts:
    ax.scatter([e], [a], s=140, color=c, zorder=3)
    ax.annotate(label, (e, a), textcoords="offset points", xytext=(8, -4), fontsize=10)

# arrow: QAT lifts int4 back up at the same energy
ax.annotate("", xy=(E4, 91.62), xytext=(E4, 87.79),
            arrowprops=dict(arrowstyle="->", color="tab:green"))
ax.text(E4 - 0.35, (87.79 + 91.62) / 2, "QAT\n+3.8 pts", color="tab:green",
        ha="right", va="center", fontsize=9)

ax.set_xlabel("energy per inference (uJ, 45nm)")
ax.set_ylabel("top-1 accuracy (%)  [speech_commands test]")
ax.set_title("Accuracy-energy trade: int4+QAT ~halves energy\nfor <1% accuracy loss vs int8")
ax.set_xlim(6, 19)
ax.set_ylim(86, 93)
ax.grid(True, alpha=0.3)
plt.tight_layout()
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "fig_accuracy_energy.png")
plt.savefig(out, dpi=140)
print("wrote", os.path.relpath(out))
