// mac_array_8x8.sv
// 8x8 output-stationary int8 MAC array = the accelerator's compute core, the
// 64 `intmac` units the Timeloop model costed. Computes one 8x8 output tile
//     O[i][j] = sum_k A[i][k] * W[k][j]
// by streaming one k per enabled cycle: on cycle k the array is fed column
// A[:,k] (a_col) and row W[k,:] (w_row); PE(i,j) accumulates a_col[i]*w_row[j].
// Pointwise (1x1) conv is exactly this matmul (C_in x C_out), i.e. the layer the
// roofline flagged as compute-bound -- the natural workload for this core.
//
// Ports are flattened to packed buses so the module is portable through Yosys.
module mac_array_8x8 #(parameter int N = 8, parameter int ACCW = 32) (
    input  logic                    clk,
    input  logic                    rst,
    input  logic                    clr,      // clear all accumulators
    input  logic                    en,       // accumulate this cycle
    input  logic [N*8-1:0]          a_col,    // A[:,k] : N packed int8, a_col[i]
    input  logic [N*8-1:0]          w_row,    // W[k,:] : N packed int8, w_row[j]
    output logic [N*N*ACCW-1:0]     o_flat    // O[i][j] packed row-major
);
    genvar i, j;
    generate
        for (i = 0; i < N; i++) begin : row
            for (j = 0; j < N; j++) begin : col
                logic signed [ACCW-1:0] acc_ij;
                mac_int8 #(.ACCW(ACCW)) pe (
                    .clk (clk),
                    .rst (rst),
                    .clr (clr),
                    .en  (en),
                    .a   (a_col[i*8 +: 8]),
                    .w   (w_row[j*8 +: 8]),
                    .acc (acc_ij)
                );
                assign o_flat[(i*N + j)*ACCW +: ACCW] = acc_ij;
            end
        end
    endgenerate
endmodule
