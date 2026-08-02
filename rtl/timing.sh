#!/usr/bin/env bash
# Gate-level static timing (critical path -> Fmax) of one MAC PE in sky130.
set -e
cd "$(dirname "$0")"
LIB=sky130_fd_sc_hd__tt_025C_1v80.lib
[ -f "$LIB" ] || curl -sL -o "$LIB" "https://raw.githubusercontent.com/The-OpenROAD-Project/OpenROAD-flow-scripts/master/flow/platforms/sky130hd/lib/$LIB"
printf 'strash\ndch\nmap\ntopo\nstime -p\n' > /tmp/abc_pe.script
printf 'read_verilog -sv mac_int8.sv\nsynth -top mac_int8 -flatten\ndfflibmap -liberty %s\nabc -liberty %s -script /tmp/abc_pe.script\n' "$LIB" "$LIB" > /tmp/pe.ys
yosys -s /tmp/pe.ys 2>&1 | tee /tmp/pe.log | grep -iE "Delay =" | tail -1
