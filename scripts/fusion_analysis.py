#!/usr/bin/env python3
"""
Depthwise-aware optimization: FUSE depthwise -> pointwise.

The roofline (fig1) showed depthwise is memory-bound: its cost is data movement,
not compute. In a DS-CNN, every depthwise is immediately followed by a pointwise,
and the depthwise's OUTPUT is exactly the pointwise's INPUT. Unfused, that
intermediate activation gets written to DRAM (after depthwise) and read back from
DRAM (before pointwise) -- pure overhead. Fusing the two layers keeps it on-chip
in the global buffer instead.

This is an ENERGY optimization: the layers are compute-bound in *cycles* (the
mapper fills the array fine) but DRAM-dominated in *energy*. We model the saving
analytically using the real Accelergy ERT energies from a swept run -- no new
mapper run needed. DRAM per-access energy is ~45x the on-chip buffer, so removing
the round-trip is worth a lot.
"""
import os, glob, csv
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def kit_root():
    here = (os.path.dirname(os.path.abspath(__file__))
            if "__file__" in globals() else os.getcwd())
    for c in (here, os.path.join(here, ".."), os.getcwd()):
        if os.path.exists(os.path.join(c, "results.csv")):
            return os.path.abspath(c)
    return os.path.abspath(os.path.join(here, ".."))
ROOT = kit_root()

# --- 1. real per-access energies from an Accelergy ERT ----------------------
# DRAM and global_buffer are both width=64b, datawidth=8b -> 8 int8 words/access,
# so per-scalar energy = per-access / 8. (Verified: pointwise DRAM footprint
# 20096 elems / 8 * 512 pJ = 1.286 uJ == its per_component_energy['DRAM'].)
BLOCK = 8
_all_ert = glob.glob(os.path.join(ROOT, "runs", "*", "timeloop-mapper.ERT_summary.yaml"))
assert _all_ert, "no ERT_summary.yaml under runs/ -- run a sweep first"
# Use the CHOSEN design point's ERT (8x8/glb2048) so buffer energy is consistent.
ert_files = [f for f in _all_ert if "8x8_glb2048" in f] or _all_ert
ert = yaml.safe_load(open(ert_files[0]))

def _iter_tables(node):
    """Yield every {name, actions:[...]} table anywhere in the ERT tree."""
    if isinstance(node, dict):
        if isinstance(node.get("name"), str) and isinstance(node.get("actions"), list):
            yield node
        for v in node.values():
            yield from _iter_tables(v)
    elif isinstance(node, list):
        for v in node:
            yield from _iter_tables(v)

def action_energy(component_substr, action):
    for tbl in _iter_tables(ert):
        if component_substr in tbl["name"]:
            for a in tbl["actions"]:
                if isinstance(a, dict) and a.get("name") == action and "energy" in a:
                    return a["energy"]          # pJ per full-width access
    raise KeyError(f"{component_substr}/{action} not in ERT")

E_dram = (action_energy("DRAM", "read") + action_energy("DRAM", "write")) / BLOCK
E_glb  = (action_energy("global_buffer", "read") + action_energy("global_buffer", "write")) / BLOCK
print(f"ERT source: {os.path.relpath(ert_files[0], ROOT)}")
print(f"per-int8-element  DRAM(rd+wr) = {E_dram:.2f} pJ   GLB(rd+wr) = {E_glb:.3f} pJ "
      f"({E_dram/E_glb:.0f}x)")

# --- 2. the intermediate tensor: depthwise output == pointwise input --------
G, P, Q = 64, 25, 5
T = G * P * Q                       # 8000 int8 elements
n_blocks = 4                        # DS-CNN has 4 depthwise-separable blocks

# --- 3. fusion saving = kill 1 DRAM write + 1 DRAM read of T, per block ------
save_per_block_uJ = T * (E_dram - E_glb) / 1e6
total_save_uJ = save_per_block_uJ * n_blocks

# --- 4. baseline per-inference energy (weighted, from the swept chosen point) -
rows = list(csv.DictReader(open(os.path.join(ROOT, "results.csv"))))
w = {"conv_standard": 1, "conv_depthwise": 4, "conv_pointwise": 4, "conv_fc": 1}
sel = {r["layer"]: r for r in rows
       if r["meshX"] == "8" and r["meshY"] == "8" and r["glb_depth"] == "2048"}
baseline_uJ = sum(w[L] * float(sel[L]["energy_uJ"]) for L in w)
fused_uJ = baseline_uJ - total_save_uJ

# --- 5. capacity gate: fusion must hold T + pointwise weights on-chip --------
def glb_bytes(depth): return depth * 64 // 8       # depth rows * 64b / 8
need_bytes = T + G * G                              # intermediate + pw weights (64x64)
gate = {d: ("fits" if glb_bytes(d) >= need_bytes else "TOO SMALL") for d in (1024, 2048, 4096)}

print(f"\nintermediate tensor T (dw out == pw in) = {T} elem/block, {n_blocks} blocks")
print(f"fusion saving = {save_per_block_uJ:.3f} uJ/block  x{n_blocks} = {total_save_uJ:.3f} uJ")
print(f"per-inference energy:  unfused {baseline_uJ:.2f} uJ  ->  fused {fused_uJ:.2f} uJ "
      f"({100*total_save_uJ/baseline_uJ:.0f}% lower)")
print(f"capacity to fuse (need {need_bytes} B on-chip): "
      + ", ".join(f"glb{d}={v}" for d, v in gate.items()))
print("=> fusion REQUIRES glb>=2048; this is the co-design link that justifies the "
      "buffer size at the chosen design point.")

# --- 6. figure --------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6, 5))
bars = ax.bar(["unfused\n(dw & pw round-trip\nDRAM)", "fused\n(intermediate stays\nin GLB)"],
              [baseline_uJ, fused_uJ], color=["tab:red", "tab:green"], width=0.6)
ax.set_ylabel("energy per inference (uJ, 45nm)")
ax.set_title(f"Depthwise->pointwise fusion: {100*total_save_uJ/baseline_uJ:.0f}% energy cut\n"
             f"(kills DRAM round-trip of the {T}-elem intermediate x{n_blocks} blocks)")
for b, v in zip(bars, [baseline_uJ, fused_uJ]):
    ax.text(b.get_x() + b.get_width()/2, v, f"{v:.1f} uJ", ha="center", va="bottom")
ax.annotate("", xy=(1, fused_uJ), xytext=(1, baseline_uJ),
            arrowprops=dict(arrowstyle="<->", color="black"))
ax.text(1.08, (baseline_uJ + fused_uJ)/2, f"-{total_save_uJ:.1f} uJ", va="center")
ax.set_ylim(0, baseline_uJ * 1.15)
plt.tight_layout()
out = os.path.join(ROOT, "fig_fusion.png")
plt.savefig(out, dpi=140)
print(f"\nwrote {os.path.relpath(out, ROOT)}")
