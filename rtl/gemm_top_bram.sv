// gemm_top_bram.sv
//
// FPGA-oriented, drop-in implementation of gemm_top.  It preserves the
// byte-load/start/done/read interface but changes the three large behavioural
// arrays into N independent, single-port synchronous-read banks:
//
//   A[row][k]  -> A_bank[row % N][row / N][k]
//   W[k][col]  -> W_bank[col % N][k][col / N]
//   O[row][col]-> O_bank[col % N][row][col / N]
//
// Therefore one streamed K step uses one read from each A and W bank, rather
// than an impossible 8-read port memory.  The output tile is written one row
// at a time: each of the N O banks receives its one word for that row, so the
// former 64-write-port capture becomes eight cycles of legal one-write-port
// banking.  Operand fetches are synchronous and deliberately take three
// control cycles per K; correctness and portable BRAM inference are the goals
// of this first FPGA architecture, not peak throughput.
module gemm_top_bram #(
    parameter int N = 8, parameter int ACCW = 32,
    parameter int MAXM = 128, parameter int MAXK = 64, parameter int MAXP = 64
) (
    input  logic                              clk, rst,
    input  logic [$clog2(MAXM+1)-1:0]         m_dim,
    input  logic [$clog2(MAXK+1)-1:0]         k_dim,
    input  logic [$clog2(MAXP+1)-1:0]         p_dim,
    input  logic                              wr_en, wr_is_w,
    input  logic [$clog2(MAXM*MAXK)-1:0]      wr_addr,
    input  logic signed [7:0]                 wr_data,
    input  logic                              start,
    output logic                              busy, done,
    input  logic [$clog2(MAXM*MAXP)-1:0]      rd_addr,
    output logic signed [ACCW-1:0]            rd_data
);
    localparam int AROWS = (MAXM + N - 1) / N;
    localparam int PCOLS = (MAXP + N - 1) / N;

    // The A/W/O bank memories are declared *inside* the per-bank generate block
    // below (one 1-write/1-read RAM each), not as module-level [0:N-1][...] arrays.
    // A single 2D array indexed by the bank genvar is seen by memory_collect as one
    // memory with N write + N read ports, which cannot map to a 2-port DP16KD -- the
    // 32-bit O_mem in particular then fell back to a 256 Kbit blob of flip-flops.
    logic [N*8-1:0]          a_col_q, w_row_q;
    logic signed [ACCW-1:0]  rd_data_bank [0:N-1];
    logic                    clr, en;
    logic [N*N*ACCW-1:0]     o_flat;

    mac_array_8x8 #(.N(N), .ACCW(ACCW)) core (
        .clk(clk), .rst(rst), .clr(clr), .en(en),
        .a_col(a_col_q), .w_row(w_row_q), .o_flat(o_flat)
    );

    logic [$clog2(MAXM+1)-1:0] ti;
    logic [$clog2(MAXP+1)-1:0] tj;
    logic [$clog2(MAXK+1)-1:0] k;
    logic [$clog2(N)-1:0]      wr_row;

    typedef enum logic [3:0] {
        S_IDLE, S_CLR, S_FETCH, S_ENABLE, S_COMMIT, S_WRITE, S_NEXT, S_DONE
    } state_t;
    state_t state;

    // One physical bank per generate iteration, each holding its own 1-write/
    // 1-read RAM. Making the banks *separate memories* (rather than one array
    // indexed by the genvar) is what lets memory_libmap map every bank -- A, W and
    // the 32-bit O alike -- to its own DP16KD block RAM.
    genvar b;
    generate
        for (b = 0; b < N; b++) begin : banks
            logic signed [7:0]      A_mem [0:AROWS*MAXK-1];   // bank of A (row%N==b)
            logic signed [7:0]      W_mem [0:MAXK*PCOLS-1];   // bank of W (col%N==b)
            logic signed [ACCW-1:0] O_mem [0:MAXM*PCOLS-1];   // bank of O (col%N==b)

            // A bank: 1 write (byte load, while idle) + 1 sync read (operand fetch).
            always_ff @(posedge clk) begin
                if (state == S_IDLE && wr_en && !wr_is_w &&
                    ((wr_addr / MAXK) % N) == b)
                    A_mem[((wr_addr / MAXK) / N) * MAXK + (wr_addr % MAXK)] <= wr_data;
                else if (state == S_FETCH)
                    a_col_q[b*8 +: 8] <= ((ti + b) < m_dim)
                                       ? A_mem[(ti / N) * MAXK + k] : 8'sd0;
            end

            // W bank: 1 write (byte load) + 1 sync read (operand fetch).
            always_ff @(posedge clk) begin
                if (state == S_IDLE && wr_en && wr_is_w &&
                    ((wr_addr % MAXP) % N) == b)
                    W_mem[(wr_addr / MAXP) * PCOLS + ((wr_addr % MAXP) / N)] <= wr_data;
                else if (state == S_FETCH)
                    w_row_q[b*8 +: 8] <= ((tj + b) < p_dim)
                                       ? W_mem[k * PCOLS + (tj / N)] : 8'sd0;
            end

            // O bank: 1 write (tile writeback, one array row per cycle) + 1 sync
            // read (result readback). Kept as two processes with an unconditional
            // read -- the canonical single-clock SDP shape for DP16KD inference.
            always_ff @(posedge clk)
                if (state == S_WRITE && (ti + wr_row) < m_dim && (tj + b) < p_dim)
                    O_mem[(ti + wr_row) * PCOLS + (tj / N)] <=
                        o_flat[(wr_row*N + b)*ACCW +: ACCW];
            always_ff @(posedge clk)
                rd_data_bank[b] <= O_mem[(rd_addr / MAXP) * PCOLS +
                                            ((rd_addr % MAXP) / N)];
        end
    endgenerate

    always_comb rd_data = rd_data_bank[rd_addr % N];

    always_ff @(posedge clk) begin
        if (rst) begin
            state <= S_IDLE; ti <= '0; tj <= '0; k <= '0; wr_row <= '0;
            clr <= 1'b0; en <= 1'b0; busy <= 1'b0; done <= 1'b0;
        end else begin
            clr <= 1'b0;
            en  <= 1'b0;
            case (state)
                S_IDLE: begin
                    done <= 1'b0;
                    if (start) begin
                        busy <= 1'b1;
                        ti <= '0; tj <= '0; k <= '0;
                        state <= S_CLR;
                    end
                end
                // clr/en are registers feeding the MAC array, hence the
                // following state is where the array observes each pulse.
                S_CLR:    begin clr <= 1'b1; state <= S_FETCH; end
                S_FETCH:  state <= S_ENABLE;
                S_ENABLE: begin en <= 1'b1; state <= S_COMMIT; end
                S_COMMIT: begin
                    if (k == k_dim - 1) begin
                        wr_row <= '0;
                        state <= S_WRITE;
                    end else begin
                        k <= k + 1'b1;
                        state <= S_FETCH;
                    end
                end
                // One output row per cycle: bank b gets O[wr_row][b].
                S_WRITE: begin
                    if (wr_row == N - 1) state <= S_NEXT;
                    else wr_row <= wr_row + 1'b1;
                end
                S_NEXT: begin
                    if (tj + N < p_dim) begin
                        tj <= tj + N[$bits(tj)-1:0];
                        k <= '0;
                        state <= S_CLR;
                    end else begin
                        tj <= '0;
                        if (ti + N < m_dim) begin
                            ti <= ti + N[$bits(ti)-1:0];
                            k <= '0;
                            state <= S_CLR;
                        end else begin
                            state <= S_DONE;
                        end
                    end
                end
                S_DONE: begin
                    busy <= 1'b0;
                    done <= 1'b1;
                    if (!start) state <= S_IDLE;
                end
                default: state <= S_IDLE;
            endcase
        end
    end
endmodule
