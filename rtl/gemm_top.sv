// gemm_top.sv
// Phase 1 of the hardware sequencer: a tiling GEMM engine. It runs an arbitrary
// (M x K) * (K x P) int8 -> int32 matrix multiply on the verified 8x8 array by
// looping the array over 8x8 output tiles *in hardware* -- the step from
// accel_top's single tile toward a sequencer that needs no software tiling loop.
// This is the one op every DS-CNN layer lowers to (accel_sim.matmul): standard,
// pointwise, and depthwise convs (via im2col) and the FC are all tiled GEMMs.
//
// Dims (M,K,P) are runtime inputs (<= the MAX* params). Out-of-range rows/cols of
// an edge tile are fetched as 0, so partial tiles are bit-exact vs the software
// model. Buffers are behavioural memories (as in accel_top) -- a real chip would
// stream these from SRAM; here they make the datapath simulatable and checkable.
module gemm_top #(
    parameter int N = 8, parameter int ACCW = 32,
    parameter int MAXM = 128, parameter int MAXK = 64, parameter int MAXP = 64
) (
    input  logic                              clk, rst,
    // ---- current-GEMM dimensions ----
    input  logic [$clog2(MAXM+1)-1:0]         m_dim,
    input  logic [$clog2(MAXK+1)-1:0]         k_dim,
    input  logic [$clog2(MAXP+1)-1:0]         p_dim,
    // ---- operand load: A[row][kk] @ row*MAXK+kk ; W[kk][col] @ kk*MAXP+col ----
    input  logic                              wr_en, wr_is_w,
    input  logic [$clog2(MAXM*MAXK)-1:0]      wr_addr,
    input  logic signed [7:0]                 wr_data,
    // ---- control handshake ----
    input  logic                              start,
    output logic                              busy, done,
    // ---- result read: O[row][col] @ row*MAXP+col (int32) ----
    input  logic [$clog2(MAXM*MAXP)-1:0]      rd_addr,
    output logic signed [ACCW-1:0]            rd_data
);
    // ---- operand / result buffers ----------------------------------------
    logic signed [7:0]      A_mem [0:MAXM*MAXK-1];
    logic signed [7:0]      W_mem [0:MAXK*MAXP-1];
    logic signed [ACCW-1:0] O_mem [0:MAXM*MAXP-1];

    always_ff @(posedge clk) if (wr_en) begin
        if (wr_is_w) W_mem[wr_addr] <= wr_data;
        else         A_mem[wr_addr] <= wr_data;
    end
    // Registered (synchronous) read so O_mem infers Block RAM on FPGA instead of
    // LUT-RAM: rd_data is O_mem[rd_addr] one cycle later. Consumers read O with a
    // 2-cycle (address-then-data) handshake.
    always_ff @(posedge clk) rd_data <= O_mem[rd_addr];

    // ---- the compute core ------------------------------------------------
    logic                clr, en;
    logic [N*8-1:0]      a_col, w_row;
    logic [N*N*ACCW-1:0] o_flat;
    mac_array_8x8 #(.N(N), .ACCW(ACCW)) core (
        .clk(clk), .rst(rst), .clr(clr), .en(en),
        .a_col(a_col), .w_row(w_row), .o_flat(o_flat)
    );

    // ---- tile position + stream index ------------------------------------
    logic [$clog2(MAXM+1)-1:0] ti;      // tile row base (multiple of N)
    logic [$clog2(MAXP+1)-1:0] tj;      // tile col base (multiple of N)
    logic [$clog2(MAXK+1)-1:0] k;       // streamed inner index 0..k_dim-1

    // Column k of the current tile onto the array; 0 for rows/cols past the dims.
    always_comb begin
        for (int r = 0; r < N; r++)
            a_col[r*8 +: 8] = ((ti + r) < m_dim) ? A_mem[(ti + r)*MAXK + k] : 8'sd0;
        for (int c = 0; c < N; c++)
            w_row[c*8 +: 8] = ((tj + c) < p_dim) ? W_mem[k*MAXP + (tj + c)] : 8'sd0;
    end

    // ---- tiling control FSM ----------------------------------------------
    typedef enum logic [2:0] {S_IDLE, S_CLR, S_LOAD, S_STREAM, S_CAPTURE, S_NEXT, S_DONE} state_t;
    state_t state;

    always_ff @(posedge clk) begin
        if (rst) begin
            state <= S_IDLE; ti <= '0; tj <= '0; k <= '0;
            clr <= 1'b0; en <= 1'b0; busy <= 1'b0; done <= 1'b0;
        end else begin
            clr <= 1'b0; en <= 1'b0;                   // defaults each cycle
            case (state)
                S_IDLE: begin
                    done <= 1'b0;
                    if (start) begin
                        busy <= 1'b1; ti <= '0; tj <= '0; state <= S_CLR;
                    end
                end
                S_CLR:    begin clr <= 1'b1; state <= S_LOAD; end   // clear this tile's accumulators
                S_LOAD:   begin k <= '0; en <= 1'b1; state <= S_STREAM; end
                S_STREAM: begin
                    // en is registered: column k is accumulated on this cycle's edge.
                    if (k == k_dim - 1) begin
                        en <= 1'b0; state <= S_CAPTURE;   // last column: stop, capture next
                    end else begin
                        en <= 1'b1; k <= k + 1'b1;
                    end
                end
                S_CAPTURE: begin                          // o_flat holds the 8x8 tile
                    for (int r = 0; r < N; r++)
                        for (int c = 0; c < N; c++)
                            if ((ti + r) < m_dim && (tj + c) < p_dim)
                                O_mem[(ti + r)*MAXP + (tj + c)] <= o_flat[(r*N + c)*ACCW +: ACCW];
                    state <= S_NEXT;
                end
                S_NEXT: begin                             // advance to the next output tile
                    if (tj + N < p_dim) begin
                        tj <= tj + N[$bits(tj)-1:0]; state <= S_CLR;
                    end else begin
                        tj <= '0;
                        if (ti + N < m_dim) begin
                            ti <= ti + N[$bits(ti)-1:0]; state <= S_CLR;
                        end else begin
                            state <= S_DONE;
                        end
                    end
                end
                S_DONE: begin
                    busy <= 1'b0; done <= 1'b1;
                    if (!start) state <= S_IDLE;
                end
                default: state <= S_IDLE;
            endcase
        end
    end
endmodule
