// mac_array_8x8_pipe_pnr.sv
// Deeper-pipelined place-and-route top: the Step-3 answer to the post-route
// sign-off finding that mac_array_8x8_pnr's frequency was set NOT by the MAC
// arithmetic but by the single-cycle 64:1x32b accumulator read-out mux (see
// rtl/openlane/SIGNOFF.md). Two changes vs mac_array_8x8_pnr:
//   1. the compute core is mac_array_8x8_pipe (registered product -> the
//      multiply and accumulate-add sit in separate stages), and
//   2. the read-out mux is split into TWO registered stages, so the deep
//      64:1 select tree becomes two 8:1 selects.
// Interface matches mac_array_8x8_pnr. Latencies grow (accumulate +1 cycle;
// read-out is 2 cycles instead of 1) -- the accumulators are read only after
// the matmul has settled, so o_flat is stable and the extra read latency is
// just "present rd_sel, wait 2 cycles, sample rd_data".
module mac_array_8x8_pipe_pnr #(parameter int N = 8, parameter int ACCW = 32) (
    input  logic                       clk, rst, clr, en,
    input  logic [N*8-1:0]             a_col,
    input  logic [N*8-1:0]             w_row,
    input  logic [$clog2(N*N)-1:0]     rd_sel,    // which PE (0..N*N-1) to read
    output logic [ACCW-1:0]            rd_data
);
    localparam int M    = N*N;              // 64 accumulators
    localparam int SELW = $clog2(M);        // 6 : full select width
    localparam int LOW  = $clog2(N);        // 3 : within-group select
    localparam int HIW  = SELW - LOW;       // 3 : which group of N

    logic [M*ACCW-1:0] o_flat;

    mac_array_8x8_pipe #(.N(N), .ACCW(ACCW)) core (
        .clk(clk), .rst(rst), .clr(clr), .en(en),
        .a_col(a_col), .w_row(w_row), .o_flat(o_flat)
    );

    // rd_sel = { hi (which group of N), lo (which of N inside the group) }
    logic [LOW-1:0] lo;
    logic [HIW-1:0] hi;
    assign lo = rd_sel[LOW-1:0];
    assign hi = rd_sel[SELW-1:LOW];

    // Stage 1: N parallel N:1 muxes -- for each group g, pick word (g*N + lo).
    // Registers the N candidates and forwards the group index.
    logic [N*ACCW-1:0] s1_q;
    logic [HIW-1:0]    hi_q;
    integer g;
    always_ff @(posedge clk) begin
        for (g = 0; g < N; g = g + 1)
            s1_q[g*ACCW +: ACCW] <= o_flat[(g*N + lo)*ACCW +: ACCW];
        hi_q <= hi;
    end

    // Stage 2: N:1 mux across the group candidates -> registered read data.
    always_ff @(posedge clk)
        rd_data <= s1_q[hi_q*ACCW +: ACCW];
endmodule
