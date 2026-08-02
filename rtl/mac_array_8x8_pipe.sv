// mac_array_8x8_pipe.sv
// Same 8x8 output-stationary MAC array as mac_array_8x8, but built from the
// pipelined PE (mac_int8_pipe). Identical interface and dataflow; the only
// behavioural difference is a one-cycle accumulate latency (drive `en` low for
// one extra cycle after the last operand to flush the final product).
module mac_array_8x8_pipe #(parameter int N = 8, parameter int ACCW = 32) (
    input  logic                clk, rst, clr, en,
    input  logic [N*8-1:0]      a_col,
    input  logic [N*8-1:0]      w_row,
    output logic [N*N*ACCW-1:0] o_flat
);
    genvar i, j;
    generate
        for (i = 0; i < N; i++) begin : row
            for (j = 0; j < N; j++) begin : col
                logic signed [ACCW-1:0] acc_ij;
                mac_int8_pipe #(.ACCW(ACCW)) pe (
                    .clk(clk), .rst(rst), .clr(clr), .en(en),
                    .a(a_col[i*8 +: 8]), .w(w_row[j*8 +: 8]), .acc(acc_ij)
                );
                assign o_flat[(i*N + j)*ACCW +: ACCW] = acc_ij;
            end
        end
    endgenerate
endmodule
