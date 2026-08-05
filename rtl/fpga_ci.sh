#!/usr/bin/env bash
# fpga_ci.sh -- ECP5 synthesis + place-and-route fit report for the accelerator
# engines and the full multi-layer sequencer, for a CI runner (x86 Linux) with the
# OSS CAD Suite (yosys + nextpnr-ecp5) on PATH. Reports LUT/FF/BRAM/DSP and, where
# the design fits, nextpnr's Fmax -- the data that drives the FPGA-oriented rework.
# Each target is time-boxed so a synthesis-hostile design can't hang the job.
cd "$(dirname "$0")"
DEV=${DEV:-45k}; PKG=${PKG:-CABGA381}; TMO=${TMO:-1200}
ENG="mac_int8.sv mac_array_8x8.sv gemm_top.sv im2col_gen.sv requant_unit.sv"

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

synth gemm_top     "mac_int8.sv mac_array_8x8.sv gemm_top.sv"
synth layer_engine "$ENG layer_engine.sv"
synth dw_engine    "$ENG dw_engine.sv"
synth fc_engine    "mac_int8.sv mac_array_8x8.sv gemm_top.sv requant_unit.sv fc_engine.sv"
synth dscnn_seq    "$ENG layer_engine.sv dw_engine.sv fc_engine.sv dscnn_seq.sv"
echo "== done =="
