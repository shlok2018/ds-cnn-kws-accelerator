#!/usr/bin/env bash
# fpga_ci.sh -- ECP5 synthesis + place-and-route fit report for the accelerator
# engines and the full multi-layer sequencer, for a CI runner (x86 Linux) with the
# OSS CAD Suite (yosys + nextpnr-ecp5) on PATH. Reports LUT/FF/BRAM/DSP and, where
# the design fits, nextpnr's Fmax -- the data that drives the FPGA-oriented rework.
# Each target is time-boxed so a synthesis-hostile design can't hang the job.
cd "$(dirname "$0")"
DEV=${DEV:-45k}; PKG=${PKG:-CABGA381}; TMO=${TMO:-1200}
ENG="mac_int8.sv mac_array_8x8.sv gemm_top.sv im2col_gen.sv requant_unit.sv"

# NOTE (2026-08-05): accel_top -- the 8x8 int8 compute core -- maps to the ECP5
# cleanly (small 64-entry operand/result memories). It is the real "runs on FPGA
# fabric" result (~54 MHz, one DSP per PE). The large-buffer sequencer engines
# below are simulation-oriented: the parallel array writes all 64 accumulators
# per cycle (=64 O_mem write ports) and reads 8 A/W words per cycle (=8 read
# ports), which do not map to 1-2-port Block RAM without a banked/streamed
# datapath re-architecture. They are kept here (short timeout) only to document
# that; the verified functional model lives in the iverilog testbenches.

synth() {   # $1 = top, $2 = sources
  echo "======================== $1 ========================"
  if timeout "$TMO" yosys -p "read_verilog -sv $2; synth_ecp5 -top $1 -json /tmp/$1.json; stat" \
        > "/tmp/$1.syn.log" 2>&1; then
    grep -iE "Number of cells|LUT4 |TRELLIS_FF|CCU2|TRELLIS_DPR|DP16KD|MULT18" "/tmp/$1.syn.log" | tail -12
  else
    echo "  SYNTH did not finish within ${TMO}s (synthesis-hostile as-is)"; return
  fi
  if command -v nextpnr-ecp5 >/dev/null 2>&1; then
    if timeout "$TMO" nextpnr-ecp5 --"$DEV" --package "$PKG" --json "/tmp/$1.json" \
          --textcfg "/tmp/$1.cfg" --lpf-allow-unconstrained > "/tmp/$1.pnr.log" 2>&1; then
      grep -iE "Max frequency|Device utilisation|Info: Device" "/tmp/$1.pnr.log" | tail -25
    else
      echo "  nextpnr did not finish / did not fit within ${TMO}s"
      grep -iE "ERROR|overmap|not fit|Unable" "/tmp/$1.pnr.log" | tail -8
    fi
  fi
}

# The FPGA-fabric result: the compute core.
synth accel_top    "mac_int8.sv mac_array_8x8.sv accel_top.sv"

# Sequencer engines: documented as synthesis-hostile (short timeout), see note above.
TMO=120 synth dscnn_seq "$ENG layer_engine.sv dw_engine.sv fc_engine.sv dscnn_seq.sv"
echo "== done =="
