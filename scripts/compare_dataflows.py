#!/usr/bin/env python3
# ============================================================================
# compare_dataflows.py
# Quantifies the "does a systolic array idle on depthwise?" question by running
# every (array size x layer) under TWO spatial-mapping regimes:
#
#   free  (arch/baseline_8x8.yaml)      -- mapper may spread ANY dim across PEs,
#                                          including the depthwise group dim G.
#   cxm   (arch/baseline_8x8_cxm.yaml)  -- RIGID TPU-like pin: only input-channel
#                                          C and output-channel M map to the array.
#
# The delta is the co-design finding: under a rigid C x M systolic dataflow,
# depthwise (C=1,M=1 per group) cannot fill the reduction axis and the array
# starves; a flexible group-parallel mapping recovers it. glb depth is fixed so
# the ONLY variable is the array shape and the dataflow regime.
# ============================================================================

import os
import csv
import sys

from joblib import Parallel, delayed
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sweep  # reuse run_one() + its ROOT/CWD handling

ROOT       = sweep.ROOT
GLB_DEPTH  = 2048
ARRAYS     = [(4, 4), (8, 8), (8, 16), (16, 16)]
LAYERS     = ["conv_standard", "conv_depthwise", "conv_pointwise"]
REGIMES    = [("free", None), ("cxm", "arch/baseline_8x8_cxm.yaml")]
OUT_CSV    = os.path.join(ROOT, "results_dataflow.csv")
OUT_FIG    = os.path.join(ROOT, "fig_dataflow_compare.png")


def one(mx, my, layer, regime, arch):
    r = sweep.run_one(mx, my, GLB_DEPTH, layer, arch=arch, dataflow=regime)
    r["regime"] = regime
    r["array"]  = f"{mx}x{my}"
    return r


def main():
    jobs = [(mx, my, layer, regime, arch)
            for (mx, my) in ARRAYS
            for layer in LAYERS
            for (regime, arch) in REGIMES]
    print(f"running {len(jobs)} points (2 dataflow regimes x {len(ARRAYS)} arrays x {len(LAYERS)} layers)")
    rows = Parallel(n_jobs=-1, verbose=5)(delayed(one)(*j) for j in jobs)

    fields = ["regime", "array", "layer", "meshX", "meshY", "pe_utilization",
              "cycles", "energy_uJ", "macs", "ok"]
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {OUT_CSV}")

    # ---- figure: utilization (left) and cycles (right) vs array, per regime ----
    idx = {(r["regime"], r["layer"], r["array"]): r for r in rows if r.get("ok")}
    xlabels = [f"{mx}x{my}" for (mx, my) in ARRAYS]
    x = range(len(xlabels))
    colors = {"conv_standard": "tab:blue", "conv_depthwise": "tab:red",
              "conv_pointwise": "tab:green"}
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    for layer in LAYERS:
        for regime, style in [("free", "-o"), ("cxm", "--x")]:
            util = [idx[(regime, layer, a)]["pe_utilization"] for a in xlabels]
            cyc  = [idx[(regime, layer, a)]["cycles"] for a in xlabels]
            lbl = f"{layer.replace('conv_','')} ({regime})"
            ax1.plot(x, util, style, color=colors[layer], label=lbl)
            ax2.plot(x, cyc,  style, color=colors[layer], label=lbl)
    for ax, ttl, ylab in [(ax1, "PE utilization vs array size", "PE utilization (%)"),
                          (ax2, "Latency vs array size", "cycles (log)")]:
        ax.set_xticks(list(x)); ax.set_xticklabels(xlabels)
        ax.set_xlabel("array (meshX x meshY)"); ax.set_ylabel(ylab); ax.set_title(ttl)
        ax.grid(True, alpha=0.3)
    ax2.set_yscale("log")
    ax1.legend(fontsize=8, ncol=1)
    fig.suptitle("Rigid C x M systolic pin (dashed) under-utilizes ALL DS-CNN layers "
                 "(channels <=64 can't fill the array) vs flexible mapping (solid)")
    plt.tight_layout()
    plt.savefig(OUT_FIG, dpi=140)
    print(f"wrote {OUT_FIG}")

    # ---- printed slowdown table -------------------------------------------
    print("\nCycle slowdown under rigid C x M pin (cxm/free):")
    print(f"  {'layer':16} " + " ".join(f"{a:>8}" for a in xlabels))
    for layer in LAYERS:
        ratios = []
        for a in xlabels:
            f_ = idx[("free", layer, a)]["cycles"]
            c_ = idx[("cxm", layer, a)]["cycles"]
            ratios.append(f"{(c_/f_):7.1f}x" if f_ else "   n/a")
        print(f"  {layer:16} " + " ".join(f"{r:>8}" for r in ratios))


if __name__ == "__main__":
    main()
