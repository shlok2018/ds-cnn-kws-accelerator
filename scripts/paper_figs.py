#!/usr/bin/env python3
"""
Regenerate every paper figure as a TITLELESS VECTOR PDF.

Publication figures carry their description in the LaTeX caption, not baked into
the image -- so this drops the in-plot titles (keeping the informative axis
labels and annotations) and writes .pdf (razor-sharp at any zoom) instead of
.png. Reads results.csv for the two data-driven plots; the rest use the measured
scalars reported in the paper. Run in the Timeloop container (has matplotlib):
    python3 scripts/paper_figs.py
"""
import os, csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
def out(name): return os.path.join(ROOT, name)
def save(fig, name):
    fig.tight_layout(); fig.savefig(out(name)); plt.close(fig)
    print("wrote", name)

rows = [r for r in csv.DictReader(open(out("results.csv"))) if r["ok"] == "True"]
def fl(r, k): return float(r[k])

# ---- Fig 1: roofline (from results.csv @ 8x8/glb2048) ----------------------
RP = [r for r in rows if r["meshX"]=="8" and r["meshY"]=="8" and r["glb_depth"]=="2048"]
peak = 64.0                                              # GMAC/s @ 1 GHz (64 PEs)
thr  = {r["layer"]: fl(r,"macs")/fl(r,"cycles") for r in RP}
ai   = {r["layer"]: fl(r,"arith_intensity") for r in RP}
bw   = max(thr[l]/ai[l] for l in thr)
ridge = peak/bw
fig, ax = plt.subplots(figsize=(6.8, 5))
for l in thr:
    ax.scatter(ai[l], thr[l], s=80, zorder=3)
    ax.annotate(l.replace("conv_",""), (ai[l], thr[l]),
                textcoords="offset points", xytext=(6,5), fontsize=10)
xs = np.logspace(-0.4, 2, 200)
ax.plot(xs, np.minimum(peak, bw*xs), "k--", lw=1.3,
        label=f"roofline (peak {peak:.0f} GMAC/s, ~{bw:.0f} GB/s eff)")
ax.axvline(ridge, color="grey", ls=":", lw=0.9)
ax.text(ridge*1.05, peak*0.35, f"ridge\nAI={ridge:.1f}", fontsize=9, color="grey")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("operational intensity (MACs / byte)")
ax.set_ylabel("achieved throughput (GMAC/s @ 1 GHz)")
ax.legend(fontsize=9, loc="lower right"); ax.grid(True, which="both", alpha=0.3)
save(fig, "fig1_roofline.pdf")

# ---- Fig 2: energy-latency Pareto (all points, colour = array) -------------
fig, ax = plt.subplots(figsize=(6, 5))
arrays = {}
for r in rows:
    arrays.setdefault(f'{r["meshX"]}x{r["meshY"]}', []).append(r)
for arr in ["4x4","8x8","8x16","16x16"]:
    grp = arrays.get(arr, [])
    ax.scatter([fl(r,"cycles") for r in grp], [fl(r,"energy_uJ") for r in grp], label=arr, s=60)
ax.set_xlabel("latency (cycles)"); ax.set_ylabel("energy per inference (uJ)")
ax.legend(title="array"); ax.grid(True, alpha=0.3)
save(fig, "fig2_pareto.pdf")

# ---- Fig: depthwise->pointwise fusion --------------------------------------
base, fused = 17.18, 13.17
fig, ax = plt.subplots(figsize=(5.6, 4.6))
b = ax.bar(["unfused\n(dw & pw round-trip DRAM)", "fused\n(intermediate in GLB)"],
           [base, fused], color=["tab:red","tab:green"], width=0.6)
for bar,v in zip(b,[base,fused]): ax.text(bar.get_x()+bar.get_width()/2, v, f"{v:.1f} uJ", ha="center", va="bottom")
ax.annotate("", xy=(1,fused), xytext=(1,base), arrowprops=dict(arrowstyle="<->"))
ax.text(1.08,(base+fused)/2, f"-{base-fused:.1f} uJ\n({100*(base-fused)/base:.0f}%)", va="center")
ax.set_ylabel("energy per inference (uJ, 45nm)"); ax.set_ylim(0, base*1.15)
save(fig, "fig_fusion.pdf")

# ---- Fig: accuracy-energy (measured) ---------------------------------------
E8, E4, FREF = 17.18, 8.65, 92.17
pts = [("int8 (PTQ)",E8,92.29,"tab:blue"),("int4 (PTQ)",E4,87.79,"tab:red"),("int4 (QAT)",E4,91.62,"tab:green")]
fig, ax = plt.subplots(figsize=(6.4,4.6))
ax.axhline(FREF, ls=":", color="gray", lw=1)
ax.text(E8, FREF+0.15, f"float32 ref = {FREF:.1f}%", color="gray", ha="right", fontsize=8)
for lab,e,a,c in pts:
    ax.scatter([e],[a],s=140,color=c,zorder=3)
    ax.annotate(lab,(e,a),textcoords="offset points",xytext=(8,-4),fontsize=10)
ax.annotate("", xy=(E4,91.62), xytext=(E4,87.79), arrowprops=dict(arrowstyle="->",color="tab:green"))
ax.text(E4-0.35,(87.79+91.62)/2,"QAT\n+3.8 pts",color="tab:green",ha="right",va="center",fontsize=9)
ax.set_xlabel("energy per inference (uJ, 45nm)"); ax.set_ylabel("top-1 accuracy (%)  [speech_commands test]")
ax.set_xlim(6,19); ax.set_ylim(86,93); ax.grid(True,alpha=0.3)
save(fig, "fig_accuracy_energy.pdf")

# ---- Fig: int8 vs int4 (energy + area) -------------------------------------
fig,(a1,a2) = plt.subplots(1,2,figsize=(7,4.2))
for ax,(vals,ylab) in [(a1,([E8,E4],"energy per inference (uJ, 45nm)")),
                       (a2,([0.143,0.110],"area (mm^2, 45nm)"))]:
    bar = ax.bar(["int8","int4"], vals, color=["tab:blue","tab:orange"], width=0.6)
    for b_,v in zip(bar,vals): ax.text(b_.get_x()+b_.get_width()/2, v, f"{v:g}", ha="center", va="bottom")
    ax.set_ylabel(ylab); ax.set_ylim(0, max(vals)*1.2)
save(fig, "fig_precision.pdf")

# ---- Fig: timing (assumed / gate STA / post-layout / pipelined) ------------
fig, ax = plt.subplots(figsize=(7.2,4.6))
labels = ["Phase-1\nassumed","gate STA\nunpipelined","post-layout\nP&R","gate STA\npipelined (fix)"]
vals = [1000, 147, 139, 319]
b = ax.bar(labels, vals, color=["tab:gray","tab:red","tab:orange","tab:green"], width=0.62)
for bar,v in zip(b,vals): ax.text(bar.get_x()+bar.get_width()/2, v, f"{v} MHz", ha="center", va="bottom")
ax.axhline(1000, ls="--", color="tab:gray", lw=1)
ax.annotate("STA predicts\nlayout (~6%)", xy=(2,139), xytext=(1.5,470), ha="center", fontsize=8, arrowprops=dict(arrowstyle="->"))
ax.annotate("pipeline: 2.2x", xy=(3,319), xytext=(2.5,489), arrowprops=dict(arrowstyle="->"))
ax.set_ylabel("max clock frequency (MHz)"); ax.set_ylim(0,1150)
save(fig, "fig_timing.pdf")

# ---- Fig: predicted vs measured MAC-array area -----------------------------
pred45 = 71043.0; pred130 = pred45*(130/45)**2
synth = 368903.808; silicon = synth/0.60
fig, ax = plt.subplots(figsize=(8,4.8))
labels = ["predicted 45nm\n(Accelergy)","predicted\nscaled to 130nm","measured synth\n130nm","measured P&R est\n130nm"]
vals = [pred45/1e3, pred130/1e3, synth/1e3, silicon/1e3]
b = ax.bar(labels, vals, color=["#9ecae1","tab:blue","tab:orange","#fdae6b"], width=0.65)
for bar,v in zip(b,vals): ax.text(bar.get_x()+bar.get_width()/2, v, f"{v:.0f}k", ha="center", va="bottom")
ax.set_ylabel("MAC-array area (10^3 um^2)"); ax.set_ylim(0, max(vals)*1.18)
save(fig, "fig_rtl_vs_pred.pdf")
print("done -- 7 titleless vector PDFs")
