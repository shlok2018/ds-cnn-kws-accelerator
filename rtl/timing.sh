#!/usr/bin/env bash
# Gate-level static timing (critical path -> Fmax) of the MAC PE in sky130,
# unpipelined vs pipelined.
set -e
cd "$(dirname "$0")"
LIB=sky130_fd_sc_hd__tt_025C_1v80.lib
[ -f "$LIB" ] || curl -sL -o "$LIB" "https://raw.githubusercontent.com/The-OpenROAD-Project/OpenROAD-flow-scripts/master/flow/platforms/sky130hd/lib/$LIB"
printf 'strash\ndch\nmap\ntopo\nstime\n' > /tmp/abc_pe.script
for top in mac_int8 mac_int8_pipe; do
  printf 'read_verilog -sv %s.sv\nsynth -top %s -flatten\ndfflibmap -liberty %s\nabc -liberty %s -script /tmp/abc_pe.script\n' "$top" "$top" "$LIB" "$LIB" > /tmp/t.ys
  printf '%-16s ' "$top:"; yosys -s /tmp/t.ys 2>&1 | grep -iE "Delay =" | tail -1
done
