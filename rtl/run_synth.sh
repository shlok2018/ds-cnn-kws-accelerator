#!/usr/bin/env bash
# Synthesize the MAC array to SkyWater sky130_fd_sc_hd (130nm) and report area.
# Needs yosys (brew install yosys) and curl. The liberty (~13 MB) is fetched on
# first run and gitignored.
set -e
cd "$(dirname "$0")"
LIB=sky130_fd_sc_hd__tt_025C_1v80.lib
if [ ! -f "$LIB" ]; then
  echo "fetching sky130hd liberty (~13 MB)..."
  curl -sL -o "$LIB" \
    "https://raw.githubusercontent.com/The-OpenROAD-Project/OpenROAD-flow-scripts/master/flow/platforms/sky130hd/lib/$LIB"
fi
mkdir -p reports
yosys -s synth.ys | tee /tmp/synth_full.log | awk '/Printing statistics/{f=1} f' > reports/sky130_synth.rpt
grep -iE "Chip area|sequential elements" reports/sky130_synth.rpt
