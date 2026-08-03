// mac_array_8x8_pnr.sv
// Place-and-route wrapper around the verified mac_array_8x8 core. The core
// exposes all N*N*32 = 2048 accumulator bits as one flat port, which is fine for
// simulation but would need ~2180 chip pins -- an unrealistic, pad-limited die.
// A real accelerator streams results out through a narrow port, so here we add a
// registered read-out mux: `rd_sel` picks one PE's 32-bit accumulator onto
// `rd_data`. This is the top level actually taken through OpenLane, so the
// post-layout area/timing/power reflect a logic-limited design, not a pad ring.
module mac_array_8x8_pnr #(parameter int N = 8, parameter int ACCW = 32) (
    input  logic                       clk, rst, clr, en,
    input  logic [N*8-1:0]             a_col,
    input  logic [N*8-1:0]             w_row,
    input  logic [$clog2(N*N)-1:0]     rd_sel,    // which PE (0..N*N-1) to read
    output logic [ACCW-1:0]            rd_data
);
    logic [N*N*ACCW-1:0] o_flat;

    mac_array_8x8 #(.N(N), .ACCW(ACCW)) core (
        .clk(clk), .rst(rst), .clr(clr), .en(en),
        .a_col(a_col), .w_row(w_row), .o_flat(o_flat)
    );

    // Registered N*N:1 read-out mux (a 64:1 mux of 32-bit words).
    always_ff @(posedge clk)
        rd_data <= o_flat[rd_sel*ACCW +: ACCW];
endmodule
