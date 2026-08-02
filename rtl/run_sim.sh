#!/usr/bin/env bash
# Functional verification of the MAC array: self-checking vs a golden matmul.
# Needs Icarus Verilog (brew install icarus-verilog).
set -e
cd "$(dirname "$0")"
iverilog -g2012 -o /tmp/tb_mac.vvp mac_int8.sv mac_array_8x8.sv tb_mac_array.sv
vvp /tmp/tb_mac.vvp
