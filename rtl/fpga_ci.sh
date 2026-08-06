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
# fabric" result (~54 MHz, one DSP per PE). The large-buffer *behavioural* engines
# (gemm_top and up) are simulation-oriented: the parallel array writes all 64
# accumulators per cycle (=64 O_mem write ports) and reads 8 A/W words per cycle
# (=8 read ports), which do not map to 1-2-port Block RAM.
#
# UPDATE (2026-08-05): gemm_top_bram is the banked-BRAM re-architecture of that
# core -- A banked by row%8, W and O banked by col%8, all single-port synchronous
# banks, with the 64-wide tile writeback serialised over 8 cycles. It is bit-exact
# vs gemm_top (tb_gemm_bram) and, unlike gemm_top, maps to real DP16KD block RAM
# (A/W/O all inferred as BRAM) + 64 MULT18X18 DSPs. It is the first sequencer-path
# datapath that actually fits FPGA fabric; the fit report below quantifies it.

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

# The banked-BRAM GEMM core: the sequencer datapath re-architected to fit fabric.
synth gemm_top_bram "mac_int8.sv mac_array_8x8.sv gemm_top_bram.sv"

# UPDATE (2026-08-06): the FULL multi-layer sequencer now maps to fabric. The
# engines were re-architected -- banked-BRAM gemm core, ONE 8x8 MAC array shared
# across le/dw/fc (EXT_GEMM), ONE shared requant lane (EXT_RQ), wmem/bufA/bufB
# banked to BRAM, and im2col column addressing done with counters. It is bit-exact
# vs the software golden (tb_seq + 6 real-MFCC clips). On ECP5-45 it exceeds the
# 72-DSP budget (~97 DSP from the shared array + per-layer dimension products), so
# it is placed on the larger ECP5-85 (156 DSP); the LUT/BRAM/FF all fit ECP5-45.
SEQ_SRC="mac_int8.sv mac_array_8x8.sv gemm_top_bram.sv im2col_gen.sv requant_unit.sv \
         layer_engine.sv dw_engine.sv fc_engine.sv dscnn_seq.sv"
DEV=85k PKG=CABGA381 TMO=1800 synth dscnn_seq "$SEQ_SRC"
echo "== done =="
