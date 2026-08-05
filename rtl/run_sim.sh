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
echo "== banked-BRAM GEMM engine (FPGA-mappable, bit-exact vs gemm_top) =="
iverilog -g2012 -o /tmp/tb_gemm_bram.vvp mac_int8.sv mac_array_8x8.sv gemm_top_bram.sv tb_gemm_bram.sv; vvp /tmp/tb_gemm_bram.vvp
echo "== requant/activation lane (int32 -> int8 fixed-point requant + ReLU) =="
iverilog -g2012 -o /tmp/tb_requant.vvp requant_unit.sv tb_requant.sv; vvp /tmp/tb_requant.vvp
echo "== im2col address generator (conv lowering + SAME padding, DS-CNN shapes) =="
iverilog -g2012 -o /tmp/tb_im2col.vvp im2col_gen.sv tb_im2col.sv; vvp /tmp/tb_im2col.vvp
echo "== single conv/pointwise layer in HW (im2col -> GEMM -> requant chained) =="
iverilog -g2012 -o /tmp/tb_layer.vvp mac_int8.sv mac_array_8x8.sv gemm_top.sv im2col_gen.sv requant_unit.sv layer_engine.sv tb_layer.sv; vvp /tmp/tb_layer.vvp
echo "== depthwise layer in HW (per-channel conv + requant, all channels) =="
iverilog -g2012 -o /tmp/tb_dw.vvp mac_int8.sv mac_array_8x8.sv gemm_top.sv im2col_gen.sv requant_unit.sv dw_engine.sv tb_dw.sv; vvp /tmp/tb_dw.vvp
echo "== FC classifier tail in HW (global avg-pool -> requant -> FC -> argmax) =="
iverilog -g2012 -o /tmp/tb_fc.vvp mac_int8.sv mac_array_8x8.sv gemm_top.sv requant_unit.sv fc_engine.sv tb_fc.sv; vvp /tmp/tb_fc.vvp
echo "== full multi-layer sequencer (conv/dw/pw/fc dispatch, whole DS-CNN chain) =="
iverilog -g2012 -o /tmp/tb_seq.vvp mac_int8.sv mac_array_8x8.sv gemm_top.sv im2col_gen.sv requant_unit.sv layer_engine.sv dw_engine.sv fc_engine.sv dscnn_seq.sv tb_seq.sv; vvp /tmp/tb_seq.vvp
