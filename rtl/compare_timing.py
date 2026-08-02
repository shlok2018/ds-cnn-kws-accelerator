#!/usr/bin/env python3
"""
Phase 3 timing: measured Fmax (yosys + ABC gate-level STA, sky130 130nm) vs the
Phase-1 1 GHz clock assumption -- diagnosis AND fix.

The meaningful limit is the PER-MAC critical path: the array's 64 MACs run in
parallel, so the register-to-register path is one PE's multiply + accumulate.

  DIAGNOSIS: unpipelined single-cycle MAC -> 6.8 ns -> ~147 MHz. The 1 GHz
             assumption was ~7x optimistic.
  FIX:       one pipeline register between the multiplier and the accumulate-adder
             splits the path -> 3.1 ns -> ~319 MHz (2.2x), for one cycle of added
             latency. Further stages (split the multiplier / the 32b adder) would
             recover more, each at a latency cost.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

UNPIPE_PS = 6816.91   # unpipelined MAC PE (rtl/reports/pe_timing.rpt)
PIPE_PS   = 3130.22   # pipelined MAC PE   (rtl/reports/pe_pipe_timing.rpt)
ASSUMED_MHZ = 1000.0  # Phase-1 predicted clock (1 GHz)

unpipe_mhz = 1e6 / UNPIPE_PS
pipe_mhz   = 1e6 / PIPE_PS

print(f"Phase-1 assumption        : {ASSUMED_MHZ:6.0f} MHz (1 GHz)")
print(f"measured, unpipelined     : {UNPIPE_PS/1000:5.2f} ns -> {unpipe_mhz:4.0f} MHz "
      f"({ASSUMED_MHZ/unpipe_mhz:.1f}x optimistic)")
print(f"measured, pipelined (fix) : {PIPE_PS/1000:5.2f} ns -> {pipe_mhz:4.0f} MHz "
      f"({pipe_mhz/unpipe_mhz:.2f}x recovery, +1 cycle latency)")
print("\n=> Diagnosis: the 1 GHz clock was unrealistic for an unpipelined int8 MAC")
print("   in 130 nm. Fix: pipeline the MAC -> 2.2x faster. Area held; the clock")
print("   assumption did not, but it is recoverable by microarchitecture, not luck.")
print("Caveat: ABC gate-level STA (no wireload/P&R); OpenSTA/OpenLane would refine.")

fig, ax = plt.subplots(figsize=(7, 4.7))
labels = ["Phase-1\nassumed", "measured\nunpipelined", "measured\npipelined (fix)"]
vals   = [ASSUMED_MHZ, unpipe_mhz, pipe_mhz]
colors = ["tab:gray", "tab:red", "tab:green"]
bars = ax.bar(labels, vals, color=colors, width=0.6)
for b, v in zip(bars, vals):
    ax.text(b.get_x()+b.get_width()/2, v, f"{v:.0f} MHz", ha="center", va="bottom")
ax.axhline(ASSUMED_MHZ, ls="--", color="tab:gray", lw=1)
ax.annotate(f"pipeline: {pipe_mhz/unpipe_mhz:.1f}x",
            xy=(2, pipe_mhz), xytext=(1.5, pipe_mhz + 180),
            arrowprops=dict(arrowstyle="->"))
ax.set_ylabel("max clock frequency (MHz)")
ax.set_title("Phase 3 timing: 1 GHz assumption ~7x optimistic (sky130 130nm),\n"
             "recovered 2.2x by pipelining the MAC")
ax.set_ylim(0, ASSUMED_MHZ * 1.15)
plt.tight_layout()
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "fig_timing.png")
plt.savefig(out, dpi=140)
print(f"\nwrote {os.path.relpath(out)}")
