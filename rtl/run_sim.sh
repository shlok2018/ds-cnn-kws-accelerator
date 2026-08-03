#!/usr/bin/env bash
# Functional verification (self-checking vs golden matmul). Needs Icarus Verilog.
set -e
cd "$(dirname "$0")"
echo "== unpipelined =="
iverilog -g2012 -o /tmp/tb_mac.vvp  mac_int8.sv      mac_array_8x8.sv      tb_mac_array.sv;      vvp /tmp/tb_mac.vvp
echo "== pipelined =="
iverilog -g2012 -o /tmp/tb_pipe.vvp mac_int8_pipe.sv mac_array_8x8_pipe.sv tb_mac_array_pipe.sv; vvp /tmp/tb_pipe.vvp
echo "== full accelerator (load / start / done / read) =="
iverilog -g2012 -o /tmp/tb_accel.vvp mac_int8.sv mac_array_8x8.sv accel_top.sv tb_accel.sv; vvp /tmp/tb_accel.vvp
