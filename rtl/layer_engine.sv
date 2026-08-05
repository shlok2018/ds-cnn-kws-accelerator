// layer_engine.sv
// Phase 4a of the hardware sequencer: one convolution/pointwise layer executed
// end-to-end in hardware by chaining the three verified primitives --
//   im2col_gen  ->  gemm_top  ->  requant_unit
// i.e. lower the feature map to columns, tile-multiply by the flattened weights,
// then per-output-channel integer-requantize to the int8 output feature map. This
// is the per-layer step the multi-layer sequencer (Phase 4b) will loop, chaining
// each layer's int8 output into the next layer's input.
//
// Load order (while idle): feature map (fm_*), weights (w_*), and per-output-
// channel requant params (pr_*: sel 0=mult,1=shift,2=bias). Pulse start; when done,
// read the int8 output feature map at out[m*Cout + oc] via rd_addr.
module layer_engine #(
    parameter int MAXFM = 8192, parameter int MAXM = 128,
    parameter int MAXK = 64,    parameter int MAXP = 64
) (
    input  logic                          clk, rst,
    // ---- layer geometry ----
    input  logic [7:0]                    H, W, C, R, S, stride, Cout, P, Q,
    input  logic signed [7:0]             pad_top, pad_left,
    input  logic                          relu,
    // ---- feature-map load -> im2col ----
    input  logic                          fm_wr_en,
    input  logic [$clog2(MAXFM)-1:0]      fm_wr_addr,
    input  logic signed [7:0]             fm_wr_data,
    // ---- weight load -> gemm W (wmat[k][oc] @ k*MAXP+oc) ----
    input  logic                          w_wr_en,
    input  logic [$clog2(MAXK*MAXP)-1:0]  w_wr_addr,
    input  logic signed [7:0]             w_wr_data,
    // ---- per-output-channel requant params ----
    input  logic                          pr_wr_en,
    input  logic [1:0]                     pr_sel,     // 0=mult 1=shift 2=bias
    input  logic [$clog2(MAXP)-1:0]        pr_addr,    // output channel
    input  logic signed [31:0]             pr_data,
    // ---- control ----
    input  logic                          start,
    output logic                          busy, done,
    // ---- output feature map read: out[m*Cout + oc] (int8) ----
    input  logic [$clog2(MAXM*MAXP)-1:0]  rd_addr,
    output logic signed [7:0]             rd_data
);
    localparam int ACCW = 32;
    logic [15:0] M_, K_;
    assign M_ = P * Q;                    // output positions
    assign K_ = R * S * C;                // lowered inner dim

    // ---- per-channel requant param memories + output buffer ----
    logic signed [31:0] mult_mem [0:MAXP-1];
    logic [5:0]         shift_mem[0:MAXP-1];
    logic signed [31:0] bias_mem [0:MAXP-1];
    logic signed [7:0]  out_mem  [0:MAXM*MAXP-1];
    always_ff @(posedge clk) if (pr_wr_en) begin
        if      (pr_sel == 2'd0) mult_mem [pr_addr] <= pr_data;
        else if (pr_sel == 2'd1) shift_mem[pr_addr] <= pr_data[5:0];
        else                     bias_mem [pr_addr] <= pr_data;
    end
    assign rd_data = out_mem[rd_addr];

    // ---- FSM state + shared counters (ca = row/m, cb = k or oc) ----
    typedef enum logic [2:0] {S_IDLE,S_IM2COL,S_COPY,S_GEMM,S_GWAIT,S_REQ,S_DONE} state_t;
    state_t state;
    logic [7:0] ca, cb;
    logic       i2c_start, gemm_start;
    logic       loading;
    assign loading = (state == S_IDLE);

    // ---- im2col_gen ----
    logic                          i2c_done, i2c_busy;
    logic [$clog2(MAXM*MAXK)-1:0]  i2c_rd_addr;
    logic signed [7:0]             i2c_rd_data;
    im2col_gen #(.MAXFM(MAXFM), .MAXM(MAXM), .MAXK(MAXK)) u_im2col (
        .clk(clk), .rst(rst), .H(H), .W(W), .C(C), .R(R), .S(S),
        .stride(stride), .P(P), .Q(Q), .pad_top(pad_top), .pad_left(pad_left),
        .wr_en(loading & fm_wr_en), .wr_addr(fm_wr_addr), .wr_data(fm_wr_data),
        .start(i2c_start), .busy(i2c_busy), .done(i2c_done),
        .rd_addr(i2c_rd_addr), .rd_data(i2c_rd_data)
    );

    // ---- gemm_top ----  (W loaded externally while idle; A copied from cols)
    logic                          g_busy, g_done;
    logic                          g_wr_en, g_wr_is_w;
    logic [$clog2(MAXM*MAXK)-1:0]  g_wr_addr;
    logic signed [7:0]             g_wr_data;
    logic [$clog2(MAXM*MAXP)-1:0]  g_rd_addr;
    logic signed [ACCW-1:0]        g_rd_data;
    gemm_top #(.N(8), .ACCW(ACCW), .MAXM(MAXM), .MAXK(MAXK), .MAXP(MAXP)) u_gemm (
        .clk(clk), .rst(rst), .m_dim(M_[$clog2(MAXM+1)-1:0]),
        .k_dim(K_[$clog2(MAXK+1)-1:0]), .p_dim(Cout[$clog2(MAXP+1)-1:0]),
        .wr_en(g_wr_en), .wr_is_w(g_wr_is_w), .wr_addr(g_wr_addr), .wr_data(g_wr_data),
        .start(gemm_start), .busy(g_busy), .done(g_done),
        .rd_addr(g_rd_addr), .rd_data(g_rd_data)
    );

    // gemm write port: external weights while idle, cols->A copy during S_COPY
    always_comb begin
        i2c_rd_addr = ca * MAXK + cb;          // cols[m][k] during copy
        g_rd_addr   = ca * MAXP + cb;          // O[m][oc]   during requant
        if (loading) begin
            g_wr_en = w_wr_en; g_wr_is_w = 1'b1; g_wr_addr = w_wr_addr; g_wr_data = w_wr_data;
        end else begin
            g_wr_en = (state == S_COPY); g_wr_is_w = 1'b0;
            g_wr_addr = ca * MAXK + cb; g_wr_data = i2c_rd_data;
        end
    end

    // ---- requant lane (combinational), driven during S_REQ ----
    logic signed [7:0] req_y;
    requant_unit #(.ACCW(ACCW), .OW(8)) u_req (
        .acc(g_rd_data), .bias(bias_mem[cb]), .mult(mult_mem[cb]),
        .shift(shift_mem[cb]), .relu(relu), .y(req_y)
    );

    always_ff @(posedge clk) begin
        if (rst) begin
            state <= S_IDLE; ca <= '0; cb <= '0;
            i2c_start <= 1'b0; gemm_start <= 1'b0; busy <= 1'b0; done <= 1'b0;
        end else begin
            i2c_start <= 1'b0; gemm_start <= 1'b0;      // one-cycle pulses
            case (state)
                S_IDLE: begin
                    done <= 1'b0;
                    if (start) begin busy <= 1'b1; i2c_start <= 1'b1; state <= S_IM2COL; end
                end
                S_IM2COL: if (i2c_done) begin ca <= '0; cb <= '0; state <= S_COPY; end
                S_COPY: begin                              // cols -> gemm A_mem
                    if (cb == K_ - 1) begin
                        cb <= '0;
                        if (ca == M_ - 1) begin ca <= '0; state <= S_GEMM; end
                        else ca <= ca + 8'd1;
                    end else cb <= cb + 8'd1;
                end
                S_GEMM:  begin gemm_start <= 1'b1; state <= S_GWAIT; end
                S_GWAIT: if (g_done) begin ca <= '0; cb <= '0; state <= S_REQ; end
                S_REQ: begin                               // O -> requant -> out
                    out_mem[ca * int'(Cout) + cb] <= req_y;  // int' widens: ca*Cout must not wrap at 8b
                    if (cb == Cout - 1) begin
                        cb <= '0;
                        if (ca == M_ - 1) begin ca <= '0; state <= S_DONE; end
                        else ca <= ca + 8'd1;
                    end else cb <= cb + 8'd1;
                end
                S_DONE: begin busy <= 1'b0; done <= 1'b1; if (!start) state <= S_IDLE; end
                default: state <= S_IDLE;
            endcase
        end
    end
endmodule
