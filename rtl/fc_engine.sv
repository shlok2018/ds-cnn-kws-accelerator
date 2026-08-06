// fc_engine.sv
// The DS-CNN's classifier tail in hardware: global average pool -> requant ->
// fully-connected matmul -> per-class scale/bias -> argmax. Mirrors the Dense
// branch of run_dscnn_on_accel.infer:
//   v[c]   = mean_m fmap[m][c]                          (global average pool)
//   vq[c]  = requant(sum_m fmap[m][c])                  (pool scale folds in 1/M)
//   raw[o] = sum_c vq[c] * fcw[c][o]                    (FC matmul on the array)
//   logit[o] = raw[o]*fc_mult[o] + fc_bias[o]           (per-class dequant)
//   pred   = argmax_o logit[o]
// The per-class mult/bias are integers (calibration folds the common output scale
// in), so argmax over the int64 logits equals argmax over the real logits.
module fc_engine #(
    parameter int MAXIN = 8192, parameter int MAXC = 64, parameter int MAXNC = 16,
    parameter int MAXM = 128,   parameter int MAXK = 64, parameter int MAXP = 64,
    // EXT_GEMM=1 -> use a shared external gemm via xg_* (see layer_engine).
    parameter int EXT_GEMM = 0,
    parameter int EXT_RQ = 0
) (
    input  logic                          clk, rst,
    input  logic [7:0]                    M, C, NC,        // positions, channels, classes
    // input feature-map load: fmap[m*C + c]
    input  logic                          fm_wr_en,
    input  logic [$clog2(MAXIN)-1:0]      fm_wr_addr,
    input  logic signed [7:0]             fm_wr_data,
    // FC weight load -> gemm W: fcw[c][o] @ c*MAXP + o
    input  logic                          w_wr_en,
    input  logic [$clog2(MAXK*MAXP)-1:0]  w_wr_addr,
    input  logic signed [7:0]             w_wr_data,
    // pool requant (single, per-tensor)
    input  logic signed [31:0]            pool_mult,
    input  logic [5:0]                    pool_shift,
    // per-class FC output params (sel 0=mult, 1=bias)
    input  logic                          fcp_wr_en,
    input  logic                          fcp_sel,
    input  logic [$clog2(MAXNC)-1:0]      fcp_addr,
    input  logic signed [31:0]            fcp_data,
    // control
    input  logic                          start,
    output logic                          busy, done,
    // read pooled+requant vector vq[c], and the predicted class
    input  logic [$clog2(MAXC)-1:0]       rd_addr,
    output logic signed [7:0]             vq_data,
    output logic [3:0]                    pred,
    // ---- shared external GEMM interface (only used when EXT_GEMM=1) ----
    output logic [$clog2(MAXM+1)-1:0]     xg_m_dim,
    output logic [$clog2(MAXK+1)-1:0]     xg_k_dim,
    output logic [$clog2(MAXP+1)-1:0]     xg_p_dim,
    output logic                          xg_wr_en, xg_wr_is_w,
    output logic [$clog2(MAXM*MAXK)-1:0]  xg_wr_addr,
    output logic signed [7:0]             xg_wr_data,
    output logic                          xg_start,
    output logic [$clog2(MAXM*MAXP)-1:0]  xg_rd_addr,
    input  logic                          xg_busy, xg_done,
    input  logic signed [31:0]            xg_rd_data,
    // ---- shared external requant interface (only used when EXT_RQ=1) ----
    output logic signed [31:0]            xr_acc, xr_bias, xr_mult,
    output logic [5:0]                    xr_shift,
    output logic                          xr_relu,
    input  logic signed [7:0]             xr_y
);
    localparam int ACCW = 32;
    logic loading;

    logic signed [7:0]  fmap    [0:MAXIN-1];
    logic signed [7:0]  vq      [0:MAXC-1];
    logic signed [31:0] fc_mult [0:MAXNC-1];
    logic signed [31:0] fc_bias [0:MAXNC-1];
    assign vq_data = vq[rd_addr];

    always_ff @(posedge clk) begin
        if (loading & fm_wr_en) fmap[fm_wr_addr] <= fm_wr_data;
        if (loading & fcp_wr_en) begin
            if (!fcp_sel) fc_mult[fcp_addr] <= fcp_data;
            else          fc_bias[fcp_addr] <= fcp_data;
        end
    end

    // ---- state + counters ----
    typedef enum logic [2:0] {S_IDLE,S_POOL,S_LOADA,S_GEMM,S_GWAIT,S_FA,S_FCOUT,S_DONE} state_t;
    state_t state;
    assign loading = (state == S_IDLE);
    logic [7:0]  cc, mm;
    logic [31:0] v_sum;
    logic        gemm_start;

    // running pool sum including the current element (avoids an extra cycle)
    logic signed [31:0] pool_next;
    always_comb pool_next = v_sum + fmap[mm*int'(C) + cc];

    logic signed [7:0] vqv;
    assign xr_acc = pool_next; assign xr_bias = 32'sd0; assign xr_mult = pool_mult;
    assign xr_shift = pool_shift; assign xr_relu = 1'b1;
    generate if (EXT_RQ == 0) begin : rq_local
        requant_unit #(.ACCW(ACCW),.OW(8)) u_pool (
            .acc(xr_acc),.bias(xr_bias),.mult(xr_mult),.shift(xr_shift),.relu(xr_relu),.y(vqv));
    end else begin : rq_ext
        assign vqv = xr_y;
    end endgenerate

    // ---- FC matmul on the array (1 x C)*(C x NC) ----
    logic g_busy, g_done, g_wr_en, g_wr_is_w;
    logic [$clog2(MAXM*MAXK)-1:0] g_wr_addr; logic signed [7:0] g_wr_data;
    logic [$clog2(MAXM*MAXP)-1:0] g_rd_addr; logic signed [ACCW-1:0] g_rd_data;
    assign xg_m_dim   = 8'd1;
    assign xg_k_dim   = C[$clog2(MAXK+1)-1:0];
    assign xg_p_dim   = NC[$clog2(MAXP+1)-1:0];
    assign xg_wr_en   = g_wr_en;   assign xg_wr_is_w = g_wr_is_w;
    assign xg_wr_addr = g_wr_addr; assign xg_wr_data = g_wr_data;
    assign xg_start   = gemm_start; assign xg_rd_addr = g_rd_addr;
    generate if (EXT_GEMM == 0) begin : g_local
        gemm_top_bram #(.N(8),.ACCW(ACCW),.MAXM(MAXM),.MAXK(MAXK),.MAXP(MAXP)) u_gemm (
            .clk(clk),.rst(rst),.m_dim(xg_m_dim),.k_dim(xg_k_dim),.p_dim(xg_p_dim),
            .wr_en(xg_wr_en),.wr_is_w(xg_wr_is_w),.wr_addr(xg_wr_addr),.wr_data(xg_wr_data),
            .start(xg_start),.busy(g_busy),.done(g_done),.rd_addr(xg_rd_addr),.rd_data(g_rd_data));
    end else begin : g_ext
        assign g_busy = xg_busy; assign g_done = xg_done; assign g_rd_data = xg_rd_data;
    end endgenerate

    always_comb begin
        g_rd_addr = cc;                                   // O[0][oc]
        if (loading)              begin g_wr_en=w_wr_en; g_wr_is_w=1'b1; g_wr_addr=w_wr_addr; g_wr_data=w_wr_data; end
        else if (state==S_LOADA)  begin g_wr_en=1'b1;    g_wr_is_w=1'b0; g_wr_addr=cc;        g_wr_data=vq[cc];   end
        else                      begin g_wr_en=1'b0;    g_wr_is_w=1'b0; g_wr_addr='0;        g_wr_data='0;       end
    end

    // per-class logit + running argmax
    logic signed [63:0] logit, best;

    always_ff @(posedge clk) begin
        if (rst) begin
            state<=S_IDLE; cc<='0; mm<='0; v_sum<='0; gemm_start<=1'b0;
            busy<=1'b0; done<=1'b0; pred<='0; best<='0;
        end else begin
            gemm_start<=1'b0;
            case (state)
                S_IDLE: begin done<=1'b0;
                    if (start) begin busy<=1'b1; cc<='0; mm<='0; v_sum<='0; state<=S_POOL; end
                end
                S_POOL: begin                              // v_sum[cc] += fmap[m][cc]; requant at last m
                    v_sum <= pool_next;
                    if (mm == M-1) begin
                        vq[cc] <= vqv; v_sum <= '0; mm <= '0;
                        if (cc == C-1) begin cc<='0; state<=S_LOADA; end else cc<=cc+8'd1;
                    end else mm <= mm + 8'd1;
                end
                S_LOADA: begin                             // vq -> gemm A row 0
                    if (cc == C-1) begin cc<='0; state<=S_GEMM; end else cc<=cc+8'd1;
                end
                S_GEMM:  begin gemm_start<=1'b1; state<=S_GWAIT; end
                S_GWAIT: if (g_done) begin cc<='0; best<=64'sh8000000000000000; state<=S_FA; end
                S_FA: state<=S_FCOUT;                      // present O[0][cc] addr; data (BRAM) next cycle
                S_FCOUT: begin                             // logit[oc] = raw*mult+bias ; argmax
                    logit = $signed(g_rd_data) * $signed(fc_mult[cc]) + $signed(fc_bias[cc]);
                    if (logit > best) begin best <= logit; pred <= cc[3:0]; end
                    if (cc == NC-1) state <= S_DONE; else begin cc <= cc + 8'd1; state <= S_FA; end
                end
                S_DONE: begin busy<=1'b0; done<=1'b1; if (!start) state<=S_IDLE; end
                default: state<=S_IDLE;
            endcase
        end
    end
endmodule
