#!/usr/bin/env python3
"""
Phase 3 timing: measured Fmax (yosys + ABC gate-level STA, sky130 130nm) vs the
Phase-1 1 GHz clock assumption.

The meaningful limit is the PER-MAC critical path: the array's 64 MACs run in
parallel, so the register-to-register path is one PE's multiply + accumulate.
(The array-level unbuffered number is ~64 ns, but that is a high-fanout broadcast
artifact of synthesis-without-buffering that place-and-route resolves -- it is not
the architectural limit, so we headline the per-MAC number and note the array
value as the P&R-fixable pessimistic bound.)
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PE_PS  = 6816.91     # single MAC PE, delay-mapped   (rtl/reports/pe_timing.rpt)
ARR_PS = 64687.31    # 8x8 array, unbuffered broadcast (fanout-limited)
ASSUMED_MHZ = 1000.0 # Phase-1 predicted clock (1 GHz)

pe_mhz  = 1e6 / PE_PS
arr_mhz = 1e6 / ARR_PS

print(f"per-MAC critical path : {PE_PS/1000:5.2f} ns  -> Fmax {pe_mhz:6.0f} MHz")
print(f"array (unbuffered)    : {ARR_PS/1000:5.1f} ns  -> Fmax {arr_mhz:6.0f} MHz  (P&R-fixable)")
print(f"Phase-1 assumption    : {ASSUMED_MHZ:6.0f} MHz (1 GHz)")
print(f"\n=> the 1 GHz clock is ~{ASSUMED_MHZ/pe_mhz:.1f}x optimistic for an unpipelined")
print("   single-cycle int8 MAC in sky130 (130 nm). The AREA prediction held to")
print("   ~1x, but the CLOCK (hence latency/throughput) did NOT -- de-rate it, or")
print("   pipeline the MAC (register the multiplier output) to recover frequency.")
print("Caveat: ABC gate-level STA, no wireload/P&R; a real OpenSTA/OpenLane sign-off")
print("   would refine both numbers (per-MAC up a little, array down a lot).")

fig, ax = plt.subplots(figsize=(6.5, 4.6))
labels = ["Phase-1\nassumed", "measured\nper-MAC", "measured array\n(unbuffered, P&R-fixable)"]
vals   = [ASSUMED_MHZ, pe_mhz, arr_mhz]
colors = ["tab:gray", "tab:green", "#c7c7c7"]
bars = ax.bar(labels, vals, color=colors, width=0.6)
for b, v in zip(bars, vals):
    ax.text(b.get_x()+b.get_width()/2, v, f"{v:.0f} MHz", ha="center", va="bottom")
ax.axhline(ASSUMED_MHZ, ls="--", color="tab:gray", lw=1)
ax.set_ylabel("max clock frequency (MHz)")
ax.set_title(f"Phase 3 timing: 1 GHz assumption is ~{ASSUMED_MHZ/pe_mhz:.0f}x optimistic\n"
             f"(sky130 130nm, unpipelined int8 MAC) -- pipeline or de-rate")
ax.set_ylim(0, ASSUMED_MHZ*1.15)
plt.tight_layout()
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "fig_timing.png")
plt.savefig(out, dpi=140)
print(f"\nwrote {os.path.relpath(out)}")
