#!/usr/bin/env python3
# ============================================================================
# analyze.py
# Turns results.csv into the five figures the Phase 1 notebook must show.
# Written as a script but structured in cells (# %%) so you can paste it
# straight into Jupyter and run block by block.
#
# The five deliverables (from the Phase 1 spec):
#   1. Workload characterization + roofline
#   2. Pareto frontier (energy vs latency), colored by array size
#   3. Per-layer-type utilization breakdown   <- the depthwise story
#   4. Chosen design point + written justification
#   5. Predictions, committed BEFORE any RTL   <- the thing people skip
# ============================================================================

# %%
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Resolve the kit root so paths work no matter where this is launched from
# (repo root, scripts/, or pasted into Jupyter where __file__ is undefined).
def _kit_root():
    here = (os.path.dirname(os.path.abspath(__file__))
            if "__file__" in globals() else os.getcwd())
    for cand in (here, os.path.join(here, ".."),
                 os.getcwd(), os.path.join(os.getcwd(), "..")):
        if os.path.exists(os.path.join(cand, "results.csv")):
            return os.path.abspath(cand)
    return os.path.abspath(os.path.join(here, ".."))
ROOT = _kit_root()

df = pd.read_csv(os.path.join(ROOT, "results.csv"))
df = df[df["ok"] == True].copy()
df["array"] = df["meshX"].astype(str) + "x" + df["meshY"].astype(str)
df["pes"]   = df["meshX"] * df["meshY"]
print(df.head())

# %% ---- FIGURE 1: roofline ------------------------------------------------
# Operational intensity = MACs / bytes moved (from the DRAM operand footprints,
# so it's the algorithmic best case) on x; achieved throughput = MACs/cycle at
# 1 GHz (GMAC/s) on y. Peak-compute roof = num_PEs x 1 GHz. The DRAM-bandwidth
# roof is drawn from the EFFECTIVE bandwidth implied by the most memory-bound
# layer (empirical, so every measured point sits on/under the roof). Layers left
# of the ridge are MEMORY-BOUND -- this is where depthwise (low reuse) and the FC
# land, which is the real reason depthwise costs more than its tiny MAC count.
RP = df[(df["meshX"] == 8) & (df["meshY"] == 8) & (df["glb_depth"] == 2048)].copy()
RP["throughput"] = RP["macs"] / RP["cycles"]                 # GMAC/s @ 1 GHz
pes  = 8 * 8
peak = float(pes)                                            # GMAC/s (1 GHz)
bw   = float((RP["throughput"] / RP["arith_intensity"]).max())  # effective GB/s
ridge = peak / bw
fig, ax = plt.subplots(figsize=(6.8, 5))
for _, r in RP.iterrows():
    ax.scatter(r["arith_intensity"], r["throughput"], s=80, zorder=3)
    ax.annotate(r["layer"].replace("conv_", ""),
                (r["arith_intensity"], r["throughput"]),
                textcoords="offset points", xytext=(6, 5), fontsize=9)
xs = np.logspace(-0.4, 2, 200)
ax.plot(xs, np.minimum(peak, bw * xs), "k--", lw=1.3,
        label=f"roofline (peak {peak:.0f} GMAC/s, ~{bw:.0f} GB/s eff)")
ax.axvline(ridge, color="grey", ls=":", lw=0.9)
ax.text(ridge * 1.05, peak * 0.35, f"ridge\nAI={ridge:.1f}", fontsize=8, color="grey")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("operational intensity (MACs / byte)")
ax.set_ylabel("achieved throughput (GMAC/s @ 1 GHz)")
ax.set_title("Fig 1. Roofline @ 8x8/glb2048 -- depthwise & FC are memory-bound")
ax.legend(fontsize=8, loc="lower right"); ax.grid(True, which="both", alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(ROOT, "fig1_roofline.png"), dpi=140)

# %% ---- FIGURE 2: Pareto frontier -----------------------------------------
# Every design point as a dot; energy vs latency; color = array size.
# The Pareto-optimal points (no other point beats them on BOTH axes) are your
# real candidates. Everything above-and-right of the frontier is dominated.
fig, ax = plt.subplots(figsize=(6, 5))
for arr, grp in df.groupby("array"):
    ax.scatter(grp["cycles"], grp["energy_uJ"], label=arr, s=60)
ax.set_xlabel("latency (cycles)")
ax.set_ylabel("energy per inference (uJ)")
ax.set_title("Fig 2. Energy-latency Pareto, by array size")
ax.legend(title="array")
plt.tight_layout()
plt.savefig(os.path.join(ROOT, "fig2_pareto.png"), dpi=140)

# %% ---- FIGURE 3: the depthwise story -------------------------------------
# Utilization per layer type, grouped by array size. The expected shape:
# depthwise utilization DROPS as the array grows, while pointwise/standard
# stay high. This single chart is the most interview-valuable thing you make.
pivot = df.pivot_table(index="array", columns="layer",
                       values="pe_utilization", aggfunc="mean")
pivot = pivot.reindex(["4x4", "8x8", "8x16", "16x16"])
pivot.plot(kind="bar", figsize=(7, 4))
plt.ylabel("PE utilization")
plt.title("Fig 3. Utilization by layer type vs array size")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(os.path.join(ROOT, "fig3_utilization.png"), dpi=140)
print("\nIf depthwise utilization falls while the array grows, that's your")
print("headline: 'a bigger array is WORSE for this workload, here's the data.'")

# %% ---- FIGURE 4: pick a design point -------------------------------------
# Aggregate each design point across all three layers (weighted by how often
# each layer type runs in the real network -- edit the weights to match your
# actual DS-CNN layer counts).
layer_weights = {"conv_standard": 1, "conv_depthwise": 4, "conv_pointwise": 4,
                 "conv_fc": 1}   # MLPerf Tiny DS-CNN: 1 conv + 4 DS blocks + FC
df["w"] = df["layer"].map(layer_weights)
agg = (df.assign(we=df["energy_uJ"] * df["w"], wc=df["cycles"] * df["w"])
         .groupby(["array", "glb_depth"])
         .agg(energy=("we", "sum"), cycles=("wc", "sum"),
              area=("area_um2", "mean"))
         .reset_index())
agg["edp"] = agg["energy"] * agg["cycles"]
best = agg.sort_values("edp").head(5)
print("\nTop 5 design points by energy-delay product:")
print(best.to_string(index=False))
print("\n--> Pick from these, but justify in PROSE: maybe the EDP winner has")
print("    absurd area, and the 2nd place is 5% worse EDP at half the area.")
print("    That reasoning IS the deliverable.")

# %% ---- FIGURE 5: commit your predictions ---------------------------------
# Fill these in from your chosen design point, then git-commit this file with
# a dated message BEFORE you write a line of Verilog. This turns Phase 3 into
# a real experiment: predicted-vs-measured, not a story told after the fact.
# Design point 8x8 @ glb2048, chosen for the "small array" reason from the
# analysis: DS-CNN's <=64 channels can't fill a big array, so 8x8 keeps per-PE
# utilization high at the smallest area. Numbers are per-inference, layer-weighted
# to the MLPerf Tiny DS-CNN (std x1 / depthwise x4 / pointwise x4 / fc x1) and
# come from a stronger final-point mapper search (victory=3000), which finds a
# 100%-utilization pointwise mapping the fast sweep (victory=500) misses.
# NOTE: 8x16 @ glb2048 is marginally better on EDP at ~1.6x the area -- swap the
# design point below and re-pull from results.csv if you'd rather optimize EDP.
PREDICTIONS = {
    "design_point":            "8x8, glb_depth=2048",
    "predicted_energy_uJ":     17.09,     # weighted per-inference (Accelergy, 45nm)
    "predicted_area_um2":      143268,    # 0.143 mm^2 (Accelergy)
    "predicted_cycles":        49656,     # weighted per-inference (Timeloop, victory=3000)
    "predicted_clock_ghz":     1.0,       # target
    "date_committed":          "2026-08-02",
    "note": "45nm Accelergy estimates; Sky130 is 130nm so absolute numbers will "
            "differ, ranking should mostly hold -- compare vs post-synthesis "
            "OpenLane in Phase 3. Depthwise = true grouped conv (G=64). "
            "Per-inference = weighted sum over the four layer types "
            "(std x1, depthwise x4, pointwise x4, fc x1), final point re-optimized "
            "at mapper victory=3000.",
}
for k, v in PREDICTIONS.items():
    print(f"{k:24s}: {v}")
