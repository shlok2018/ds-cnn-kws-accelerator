#!/usr/bin/env bash
# Functional verification (self-checking vs golden matmul). Needs Icarus Verilog.
set -e
cd "$(dirname "$0")"
echo "== unpipelined =="
iverilog -g2012 -o /tmp/tb_mac.vvp  mac_int8.sv      mac_array_8x8.sv      tb_mac_array.sv;      vvp /tmp/tb_mac.vvp
echo "== pipelined =="
iverilog -g2012 -o /tmp/tb_pipe.vvp mac_int8_pipe.sv mac_array_8x8_pipe.sv tb_mac_array_pipe.sv; vvp /tmp/tb_pipe.vvp
echo "== pipelined array + 2-stage read-out mux (P&R top) =="
iverilog -g2012 -o /tmp/tb_pipe_pnr.vvp mac_int8_pipe.sv mac_array_8x8_pipe.sv mac_array_8x8_pipe_pnr.sv tb_mac_array_pipe_pnr.sv; vvp /tmp/tb_pipe_pnr.vvp
echo "== full accelerator (load / start / done / read) =="
iverilog -g2012 -o /tmp/tb_accel.vvp mac_int8.sv mac_array_8x8.sv accel_top.sv tb_accel.sv; vvp /tmp/tb_accel.vvp
echo "== tiling GEMM engine (arbitrary MxK * KxP in HW, all DS-CNN layer shapes) =="
iverilog -g2012 -o /tmp/tb_gemm.vvp mac_int8.sv mac_array_8x8.sv gemm_top.sv tb_gemm.sv; vvp /tmp/tb_gemm.vvp
