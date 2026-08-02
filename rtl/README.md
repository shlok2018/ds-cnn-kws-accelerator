# Phase 3 — RTL + synthesis (closing the predict-then-measure loop)

Synthesizable RTL for the accelerator's **compute core** — the 8×8 int8 MAC array
that the Timeloop/Accelergy model costed — functionally verified and synthesized
to SkyWater **sky130 (130 nm)** so the *measured* area can be checked against the
*predicted* (Accelergy, 45 nm) area. This is the payoff of committing predictions
pre-RTL in Phase 1: a real predicted-vs-measured experiment.

## Files
| file | what |
|---|---|
| `mac_int8.sv`      | one int8×int8 multiply-accumulate PE (output-stationary) |
| `mac_array_8x8.sv` | 8×8 array = 64 MACs; computes an 8×8 output tile of a matmul (pointwise conv) |
| `tb_mac_array.sv`  | self-checking testbench: 20 random trials vs a golden matmul |
| `synth.ys`         | yosys: synth → map to sky130hd → area report |
| `compare_area.py`  | predicted (Accelergy 45 nm) vs measured (sky130 130 nm) |
| `run_sim.sh` / `run_synth.sh` | one-command flows |
| `reports/sky130_synth.rpt`    | saved synthesis area report |

## Run
```
./run_sim.sh                 # needs iverilog -> "PASS: 20 ... trials match golden"
./run_synth.sh               # needs yosys+curl -> Chip area ~368,904 um^2
./timing.sh                  # gate-level STA -> per-MAC critical path ~6.8 ns
python3 compare_area.py      # predicted vs measured area
python3 compare_timing.py    # predicted vs measured clock
```

## Results
- **Functional:** 20/20 random 8×8 matmul trials match the golden model.
- **Area (sky130 synth):** 0.369 mm² — 64 signed multipliers + 2048 accumulator
  flops (64 PEs × 32-bit), 11% sequential.
- **Predicted vs measured AREA:** Accelergy's 45 nm estimate (71k µm²), scaled to
  130 nm (~593k) and set against the synth area with a P&R-utilization correction
  (~615k), agrees to within ~1×. The pre-RTL model's **relative** ranking is
  validated outright and its **absolute** area to a small factor once node + P&R
  are applied — exactly the "DSE is relative" caveat, now quantified.
- **Predicted vs measured TIMING:** gate-level STA (yosys+ABC, sky130) puts the
  per-MAC critical path at **6.8 ns → ~147 MHz**, so the Phase-1 **1 GHz clock
  was ~7× optimistic** for an unpipelined single-cycle int8 MAC in 130 nm. So
  area held but the clock did not: de-rate the latency/throughput numbers, or
  pipeline the MAC (register the multiplier output). This is the honest other
  half of the loop — the prediction that *missed*, and why.

## Not done yet
- **Sign-off timing / place-and-route** (OpenSTA + OpenLane) would refine the STA
  (per-MAC up slightly, the unbuffered array number down a lot) and give true
  silicon area/power.
- Only the compute core is RTL; the buffers/control are modeled, not implemented.
