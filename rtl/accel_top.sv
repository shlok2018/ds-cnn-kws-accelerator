// accel_top.sv
// A complete (if minimal) accelerator around the verified mac_array_8x8 core:
// on-chip buffers for the A and W operands and the O result, plus a control FSM
// that streams the matmul and captures the answer. This is the step from "a MAC
// array" to "an accelerator you can actually drive": load A and W, pulse `start`,
// wait for `done`, read O back. Computes O[N][N] = A[N][K] * W[K][N] (int8 -> int32).
module accel_top #(
    parameter int N = 8, parameter int K = 8, parameter int ACCW = 32
) (
    input  logic                        clk, rst,
    // ---- operand-load port: write A (wr_is_w=0) and W (wr_is_w=1) byte-by-byte
    input  logic                        wr_en,
    input  logic                        wr_is_w,
    input  logic [$clog2(N*K)-1:0]      wr_addr,
    input  logic signed [7:0]           wr_data,
    // ---- control handshake
    input  logic                        start,
    output logic                        busy,
    output logic                        done,
    // ---- result-read port: O[rd_row][rd_col] flattened as rd_addr = row*N+col
    input  logic [$clog2(N*N)-1:0]      rd_addr,
    output logic signed [ACCW-1:0]      rd_data
);
    // ---- operand / result buffers ----------------------------------------
    logic signed [7:0]    A_mem [0:N*K-1];   // A stored row-major: A[i][k] @ i*K+k
    logic signed [7:0]    W_mem [0:K*N-1];   // W stored row-major: W[k][j] @ k*N+j
    logic signed [ACCW-1:0] O_mem [0:N*N-1]; // O[i][j] @ i*N+j

    always_ff @(posedge clk) if (wr_en) begin
        if (wr_is_w) W_mem[wr_addr] <= wr_data;
        else         A_mem[wr_addr] <= wr_data;
    end
    assign rd_data = O_mem[rd_addr];

    // ---- the compute core ------------------------------------------------
    logic                 clr, en;
    logic [N*8-1:0]       a_col, w_row;
    logic [N*N*ACCW-1:0]  o_flat;
    mac_array_8x8 #(.N(N), .ACCW(ACCW)) core (
        .clk(clk), .rst(rst), .clr(clr), .en(en),
        .a_col(a_col), .w_row(w_row), .o_flat(o_flat)
    );

    // Column k of A onto a_col; row k of W onto w_row (combinational fetch).
    logic [$clog2(K)-1:0] k;
    always_comb begin
        for (int i = 0; i < N; i++) a_col[i*8 +: 8] = A_mem[i*K + k];
        for (int j = 0; j < N; j++) w_row[j*8 +: 8] = W_mem[k*N + j];
    end

    // ---- control FSM -----------------------------------------------------
    typedef enum logic [2:0] {S_IDLE, S_RESET, S_STREAM, S_CAPTURE, S_DONE} state_t;
    state_t state;

    always_ff @(posedge clk) begin
        if (rst) begin
            state <= S_IDLE; k <= '0; clr <= 1'b0; en <= 1'b0;
            busy <= 1'b0; done <= 1'b0;
        end else begin
            clr <= 1'b0; en <= 1'b0;                 // defaults each cycle
            case (state)
                S_IDLE: begin
                    done <= 1'b0;
                    if (start) begin busy <= 1'b1; clr <= 1'b1; state <= S_RESET; end
                end
                S_RESET: begin                        // accumulators cleared, start streaming
                    k <= '0; en <= 1'b1; state <= S_STREAM;
                end
                S_STREAM: begin
                    // The core samples the *registered* en, so column k is accumulated
                    // this cycle regardless; we only need to stop en for S_CAPTURE.
                    if (k == K-1) begin
                        en    <= 1'b0;                 // last column done -> stop
                        state <= S_CAPTURE;
                    end else begin
                        en <= 1'b1;                    // accumulate A[:,k]*W[k,:]
                        k  <= k + 1'b1;
                    end
                end
                S_CAPTURE: begin                       // o_flat now holds the full result
                    for (int idx = 0; idx < N*N; idx++)
                        O_mem[idx] <= o_flat[idx*ACCW +: ACCW];
                    state <= S_DONE;
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
