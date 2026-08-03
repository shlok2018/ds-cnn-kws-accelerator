#!/usr/bin/env bash
# Map the accelerator to an FPGA with the fully open-source flow.
#   Yosys alone       -> resource estimate (LUTs / DSP multipliers / FFs).
#   + nextpnr-ecp5    -> real place & route, Fmax, and an ECP5 bitstream you can
#                        load onto a board (e.g. ULX3S).
# The int8 array uses one DSP (MULT18X18) per PE, so it targets an ECP5 nicely.
# (No `set -e`: a non-matching grep must not abort the flow before nextpnr runs.)
cd "$(dirname "$0")"
SRC="mac_int8.sv mac_array_8x8.sv accel_top.sv"
TOP=accel_top
DEV=${DEV:-45k}                 # ECP5 device: 25k / 45k / 85k (64 DSPs need >=45k)
PKG=${PKG:-CABGA381}

echo "== Yosys synthesis to ECP5 (resource estimate) =="
# No -q: quiet mode hides the stat table. Keep stderr for warnings.
yosys -p "read_verilog -sv $SRC; synth_ecp5 -top $TOP -json ${TOP}.json; stat" 2>&1 \
  | grep -iE "Number of cells|LUT4|MULT18|TRELLIS_FF|CCU2|TRELLIS_DPR" || true

if command -v nextpnr-ecp5 >/dev/null 2>&1; then
  echo "== nextpnr-ecp5 place & route (Fmax) =="
  nextpnr-ecp5 --${DEV} --package "$PKG" --json ${TOP}.json --textcfg ${TOP}.cfg \
    --lpf-allow-unconstrained 2>&1 | grep -iE "Max frequency|Device utilisation|Info: Max" || true
  if command -v ecppack >/dev/null 2>&1; then
    ecppack ${TOP}.cfg ${TOP}.bit && echo "wrote ${TOP}.bit (flashable bitstream)"
  fi
else
  echo
  echo "== nextpnr-ecp5 not installed -> resource estimate only. =="
  echo "   Install the open FPGA toolchain (yosys + nextpnr-ecp5 + prjtrellis),"
  echo "   easiest via the OSS CAD Suite:"
  echo "     https://github.com/YosysHQ/oss-cad-suite-build"
  echo "   Then re-run. To flash a real board, add an LPF pin map and a small"
  echo "   host interface (UART/SPI) around the load/read ports."
fi
