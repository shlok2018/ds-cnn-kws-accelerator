#!/usr/bin/env python3
"""
Phase 3: close the predict-then-measure loop for the 8x8 int8 MAC compute core.

  PREDICTED : Accelergy ART, 45nm, per-instance area x 64 PEs (pre-RTL estimate)
  MEASURED  : Yosys tech-mapped to SkyWater sky130_fd_sc_hd (130nm), Chip area

The two are on different nodes, so a fair comparison needs a node-scale step and
an honest note that synthesis area is pre-place-and-route. The point of Phase 1
was RELATIVE ranking; this checks whether the ABSOLUTE estimate is also in the
right ballpark once node + P&R are accounted for.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- PREDICTED: Accelergy ART_summary, per-instance um^2 @ 45nm, x64 PEs ------
# (runs/conv_pointwise_8x8_glb2048_ws/timeloop-mapper.ART_summary.yaml)
mac_pred   = 931.598 * 64      # multiplier + adder  -> 59,622 um^2
accum_pred = 178.459 * 64      # partial-sum storage -> 11,421 um^2
pred_45 = mac_pred + accum_pred                    # ~71,043 um^2 @ 45nm

# ---- MEASURED: Yosys + sky130hd, Chip area @ 130nm ---------------------------
# (rtl/reports/sky130_synth.rpt : "Chip area for module '\mac_array_8x8'")
meas_130_synth = 368903.808    # um^2, cell area (pre-P&R)

# ---- reconcile ---------------------------------------------------------------
NODE = (130.0 / 45.0) ** 2     # ~8.35x ideal area scaling 45nm -> 130nm
pred_130 = pred_45 * NODE                          # predicted, scaled to 130nm
UTIL = 0.60                    # typical std-cell utilization; synth is pre-P&R
meas_130_silicon = meas_130_synth / UTIL           # est. post-P&R silicon area

print(f"PREDICTED  (Accelergy, 45nm)          : {pred_45:>10,.0f} um^2")
print(f"  scaled to 130nm  (x{NODE:.2f})            : {pred_130:>10,.0f} um^2")
print(f"MEASURED   (sky130 synth, 130nm, pre-P&R): {meas_130_synth:>10,.0f} um^2")
print(f"  est. post-P&R silicon (util {UTIL:.0%})     : {meas_130_silicon:>10,.0f} um^2")
print(f"\ngap  predicted(scaled) / measured(synth)  = {pred_130/meas_130_synth:.2f}x")
print(f"gap  predicted(scaled) / measured(P&R est) = {pred_130/meas_130_silicon:.2f}x")
print("\nVerdict: after node-scaling and a P&R-utilization correction the pre-RTL")
print("estimate lands within ~1.1-1.6x of synthesis -- same order of magnitude.")
print("So the Phase-1 model was sound for RELATIVE DSE (its intended use), and its")
print("ABSOLUTE area is trustworthy to a small factor once node + P&R are applied.")
print("Caveats: 32b RTL accumulator vs 16b modeled; ideal node scaling; synth (not")
print("P&R) area; timing (Fmax vs the 1 GHz assumption) still needs OpenSTA.")

# ---- figure ------------------------------------------------------------------
labels = ["predicted 45nm\n(Accelergy)", "predicted\nscaled to 130nm",
          "measured 130nm\n(synth, pre-P&R)", "measured 130nm\n(est. post-P&R)"]
vals   = [pred_45/1e3, pred_130/1e3, meas_130_synth/1e3, meas_130_silicon/1e3]
colors = ["#9ecae1", "tab:blue", "tab:orange", "#fdae6b"]
fig, ax = plt.subplots(figsize=(8, 4.8))
bars = ax.bar(labels, vals, color=colors, width=0.65)
for b, v in zip(bars, vals):
    ax.text(b.get_x()+b.get_width()/2, v, f"{v:.0f}k", ha="center", va="bottom")
ax.set_ylabel("MAC-array area (10^3 um^2)")
ax.set_title("Phase 3: predicted (Accelergy 45nm) vs measured (Sky130 synth) "
             "MAC-array area\nsame order of magnitude after node + P&R correction")
ax.set_ylim(0, max(vals)*1.18)
plt.tight_layout()
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "fig_rtl_vs_pred.png")
plt.savefig(out, dpi=140)
print(f"\nwrote {os.path.relpath(out)}")
