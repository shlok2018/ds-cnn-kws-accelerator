# ASIC sign-off — measured (Sky130A, OpenLane 2.3.10)

Physical-design + sign-off half of the predicted-vs-measured loop, run on
GitHub-hosted x86_64 runners via `.github/workflows/asic-signoff.yml`
(CI-reproducible equivalent of `run.sh`). Top level: `mac_array_8x8_pnr` —
the 8×8 int8 output-stationary MAC array plus a registered 64:1×32b accumulator
read-out mux. The antenna *check* is skipped (buggy lint in 2.3.10); antenna
*diode insertion* is active.

Two runs: the committed 7.0 ns / 143 MHz design point, and a relaxed 15.0 ns
closing run (clock overridden at run time; committed `config.json` unchanged).

## Physical verification (both runs)

| Check | Result |
|---|---|
| **LVS (Netgen)** | **Circuits match uniquely** ✅ |
| **DRC (Magic)** | 0 violations ✅ |
| **DRC (KLayout)** | 0 violations ✅ |
| **Detailed-route DRC** | 0 (converged over 6 iters) ✅ |
| Residual antenna violations | 126–154 nets (antenna *check* skipped; diodes inserted) |

## Area

| Metric | 7 ns run | 15 ns run |
|---|---|---|
| Std-cell area | 0.630 mm² (629,870 µm²) | 0.610 mm² (609,871 µm²) |
| Die area | 2.68 mm² | 2.68 mm² |
| Utilization | 24.0 % | 23.3 % |

Cell area matches the pre-RTL Accelergy prediction (0.63 mm²). The 4× larger die
is the deliberately loose floorplan (`FP_CORE_UTIL 25`, `PL_TARGET_DENSITY 0.35`)
needed to route the wide read-out mux; real logic footprint is the 0.6 mm² cell area.

## Multi-corner post-route STA (9 corners, real parasitics)

Committed target **7.0 ns (143 MHz)** — **does not close**; flow exits with
deferred timing errors:

| Corner | Setup WS | Hold WS |
|---|---|---|
| nom_tt (typical) | −0.35 ns (reg-to-reg +2.40, meets) | −0.08 ns (reg-to-reg meets) |
| max_ss (slow, worst) | **−6.20 ns** | −0.99 ns |
| ff (fast) | +1.79 … +1.91 ns | +0.11 ns |

Relaxed **15.0 ns (66.7 MHz)** — flow **completes** (LVS clean, DRC 0, GDS
written); setup/hold violations downgraded to warnings:

| Corner | Setup WS | Hold WS |
|---|---|---|
| nom_tt (typical) | **+5.45 ns** | +0.33 ns |
| nom_ss / min_ss / max_ss (slow) | −0.69 / −0.53 / **−0.98 ns** | +0.14 / +0.31 / **−0.05 ns** |
| ff (fast) | +7.6 … +7.9 ns | +0.11 ns |

**Interpretation**
- OpenLane completes sign-off at 15 ns / 66.7 MHz, meeting timing at the typical
  and fast corners; the slow-slow (ss) corner is ~1 ns short. Fully closing
  *every* corner needs ≈16 ns → **~62 MHz** all-corner sign-off.
- Typical-corner headroom: nom_tt has +5.45 ns at 15 ns → ~9.55 ns path → **~105 MHz
  typical**. So the design spans ~62 MHz (worst corner) to ~105 MHz (typical),
  vs the committed 143 MHz which fails post-route.
- **Critical path (verified):** accumulator FF (`o_flat`) → 64:1×32b read-out mux
  → `rd_data` FF. ~12–16 ns of AOI-compound-gate mux delay at the ss corner.
  **The read-out mux, not the MAC arithmetic, sets Fmax.** Pipelining/registering
  the read-out (or narrowing it) is the lever to recover frequency (Step 3).

## Power

| | 7 ns / 143 MHz | 15 ns / 66.7 MHz |
|---|---|---|
| Total (metric) | 0.320 W | 0.145 W |
| Total (nom_tt typical) | 0.273 W (comb 84 %) | — |

The 273 mW typical figure at 143 MHz matches the ~263 mW pre-RTL prediction;
dynamic power scales roughly with frequency at the relaxed clock.

## Bottom line
- **Step 2(a) LVS: DONE** — circuits match uniquely.
- **Step 2(b) signoff STA + power: DONE** — full 9-corner post-route STA + power.
- Honest gap: the ~139 MHz handoff estimate was post-*placement* / typical; the
  real post-*route*, multi-corner sign-off is ~62 MHz (worst) / ~105 MHz (typical),
  gated by the accumulator read-out mux.

Runs: 7 ns = GH Actions run 30863096950; 15 ns = 30867700298. Full run dirs
(GDS + reports) are uploaded as the `openlane-signoff-run` artifact on each.
