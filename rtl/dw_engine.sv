// dw_engine.sv
// Depthwise-convolution engine (extends Phase 4a to the DS-CNN's depthwise
// layers). Depthwise has no cross-channel mixing: each of C channels is an
// independent (P*Q x R*S)*(R*S x 1) matmul, exactly accel_sim.depthwise2d. The
// engine loops the same primitives over channels -- for channel ci: im2col that
// one channel (im2col_gen with C=1, cbase=ci, Cstride=C), reload the array's
// weight column with ci's R*S taps, tile-multiply, then per-position requant --
// writing the int8 output map interleaved as out[m*C + ci].
module dw_engine #(
    parameter int MAXFM = 8192, parameter int MAXM = 128, parameter int MAXK = 64,
    parameter int MAXP = 64,    parameter int MAXC = 64, parameter int MAXRS = 16
) (
    input  logic                          clk, rst,
    input  logic [7:0]                    H, W, C, R, S, stride, P, Q,
    input  logic signed [7:0]             pad_top, pad_left,
    // feature-map load -> im2col (interleaved map, C channels/pixel)
    input  logic                          fm_wr_en,
    input  logic [$clog2(MAXFM)-1:0]      fm_wr_addr,
    input  logic signed [7:0]             fm_wr_data,
    // depthwise weight load: dw[ci][k] @ ci*MAXRS + k
    input  logic                          dw_wr_en,
    input  logic [$clog2(MAXC*MAXRS)-1:0] dw_wr_addr,
    input  logic signed [7:0]             dw_wr_data,
    // per-channel requant params (0=mult,1=shift,2=bias)
    input  logic                          pr_wr_en,
    input  logic [1:0]                    pr_sel,
    input  logic [$clog2(MAXC)-1:0]       pr_addr,
    input  logic signed [31:0]            pr_data,
    // control
    input  logic                          start,
    output logic                          busy, done,
    // output read: out[m*C + ci] (int8)
    input  logic [$clog2(MAXM*MAXC)-1:0]  rd_addr,
    output logic signed [7:0]             rd_data
);
    localparam int ACCW = 32;
    logic [15:0] M_, RS_;
    assign M_  = P * Q;                   // output positions
    assign RS_ = R * S;                   // taps per channel = inner GEMM dim

    logic signed [7:0]  dw_wmem  [0:MAXC*MAXRS-1];
    logic signed [31:0] mult_mem [0:MAXC-1];
    logic [5:0]         shift_mem[0:MAXC-1];
    logic signed [31:0] bias_mem [0:MAXC-1];
    logic signed [7:0]  out_mem  [0:MAXM*MAXC-1];
    logic               loading;

    always_ff @(posedge clk) begin
        if (loading & dw_wr_en) dw_wmem[dw_wr_addr] <= dw_wr_data;
        if (loading & pr_wr_en) begin
            if      (pr_sel==2'd0) mult_mem [pr_addr] <= pr_data;
            else if (pr_sel==2'd1) shift_mem[pr_addr] <= pr_data[5:0];
            else                   bias_mem [pr_addr] <= pr_data;
        end
    end
    assign rd_data = out_mem[rd_addr];

    typedef enum logic [3:0] {S_IDLE,S_IM2COL,S_WLOAD,S_COPY,S_GEMM,S_GWAIT,S_RA,S_REQ,S_DONE} state_t;
    state_t state;
    logic [7:0] ci, ca, cb;
    logic       i2c_start, gemm_start;
    assign loading = (state == S_IDLE);

    // ---- im2col (single channel ci) ----
    logic i2c_done, i2c_busy;
    logic [$clog2(MAXM*MAXK)-1:0] i2c_rd_addr; logic signed [7:0] i2c_rd_data;
    im2col_gen #(.MAXFM(MAXFM),.MAXM(MAXM),.MAXK(MAXK)) u_im2col (
        .clk(clk),.rst(rst),.H(H),.W(W),.C(8'd1),.R(R),.S(S),.stride(stride),
        .P(P),.Q(Q),.Cstride(C),.cbase(ci),.pad_top(pad_top),.pad_left(pad_left),
        .wr_en(loading & fm_wr_en),.wr_addr(fm_wr_addr),.wr_data(fm_wr_data),
        .start(i2c_start),.busy(i2c_busy),.done(i2c_done),
        .rd_addr(i2c_rd_addr),.rd_data(i2c_rd_data));

    // ---- gemm (M x RS)*(RS x 1) ----
    logic g_busy, g_done, g_wr_en, g_wr_is_w;
    logic [$clog2(MAXM*MAXK)-1:0] g_wr_addr; logic signed [7:0] g_wr_data;
    logic [$clog2(MAXM*MAXP)-1:0] g_rd_addr; logic signed [ACCW-1:0] g_rd_data;
    gemm_top #(.N(8),.ACCW(ACCW),.MAXM(MAXM),.MAXK(MAXK),.MAXP(MAXP)) u_gemm (
        .clk(clk),.rst(rst),.m_dim(M_[$clog2(MAXM+1)-1:0]),
        .k_dim(RS_[$clog2(MAXK+1)-1:0]),.p_dim(7'd1),
        .wr_en(g_wr_en),.wr_is_w(g_wr_is_w),.wr_addr(g_wr_addr),.wr_data(g_wr_data),
        .start(gemm_start),.busy(g_busy),.done(g_done),
        .rd_addr(g_rd_addr),.rd_data(g_rd_data));

    always_comb begin
        i2c_rd_addr = ca * MAXK + cb;
        g_rd_addr   = ca * MAXP;                 // O[m][0], single output column
        if (state == S_WLOAD) begin              // reload channel ci's weight column
            g_wr_en = 1'b1; g_wr_is_w = 1'b1;
            g_wr_addr = cb * MAXP; g_wr_data = dw_wmem[ci*MAXRS + cb];
        end else if (state == S_COPY) begin      // cols -> A
            g_wr_en = 1'b1; g_wr_is_w = 1'b0;
            g_wr_addr = ca * MAXK + cb; g_wr_data = i2c_rd_data;
        end else begin g_wr_en = 1'b0; g_wr_is_w = 1'b0; g_wr_addr = '0; g_wr_data = '0; end
    end

    logic signed [7:0] req_y;
    requant_unit #(.ACCW(ACCW),.OW(8)) u_req (
        .acc(g_rd_data),.bias(bias_mem[ci]),.mult(mult_mem[ci]),
        .shift(shift_mem[ci]),.relu(1'b1),.y(req_y));

    always_ff @(posedge clk) begin
        if (rst) begin
            state<=S_IDLE; ci<='0; ca<='0; cb<='0;
            i2c_start<=1'b0; gemm_start<=1'b0; busy<=1'b0; done<=1'b0;
        end else begin
            i2c_start<=1'b0; gemm_start<=1'b0;
            case (state)
                S_IDLE: begin done<=1'b0;
                    if (start) begin busy<=1'b1; ci<='0; i2c_start<=1'b1; state<=S_IM2COL; end
                end
                S_IM2COL: if (i2c_done) begin cb<='0; state<=S_WLOAD; end
                S_WLOAD: begin                         // load RS_ weights into gemm col 0
                    if (cb == RS_-1) begin cb<='0; ca<='0; state<=S_COPY; end
                    else cb<=cb+8'd1;
                end
                S_COPY: begin                          // cols(M x RS) -> gemm A
                    if (cb == RS_-1) begin cb<='0;
                        if (ca == M_-1) begin ca<='0; state<=S_GEMM; end else ca<=ca+8'd1;
                    end else cb<=cb+8'd1;
                end
                S_GEMM:  begin gemm_start<=1'b1; state<=S_GWAIT; end
                S_GWAIT: if (g_done) begin ca<='0; state<=S_RA; end
                S_RA: state<=S_REQ;                    // present O[ca][0] addr; data (BRAM) next cycle
                S_REQ: begin                           // O[m][0] -> requant -> out[m*C+ci]
                    out_mem[ca * int'(C) + ci] <= req_y;
                    if (ca == M_-1) begin              // channel done
                        ca<='0;
                        if (ci == C-1) state<=S_DONE;
                        else begin ci<=ci+8'd1; i2c_start<=1'b1; state<=S_IM2COL; end
                    end else begin ca<=ca+8'd1; state<=S_RA; end
                end
                S_DONE: begin busy<=1'b0; done<=1'b1; if (!start) state<=S_IDLE; end
                default: state<=S_IDLE;
            endcase
        end
    end
endmodule
