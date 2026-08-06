// dscnn_seq.sv
// Full multi-layer DS-CNN sequencer. Walks a per-layer descriptor table and
// dispatches each layer to the right engine -- standard/pointwise conv
// (layer_engine), depthwise (dw_engine), or the classifier tail (fc_engine) --
// chaining the feature-map-producing layers through two ping-pong buffers. A
// layer's output map is addressed identically to the next layer's input, so
// chaining is a plain linear copy. The terminal FC layer produces `pred` (the
// predicted class) instead of a feature map.
//
// Descriptor fields (per layer): 0H 1W 2C 3R 4S 5stride 6Cout 7P 8Q 9pt 10pl
// 11relu 12woff 13poff 14type(0 conv/pw,1 dw,2 fc) 15pool_mult 16pool_shift.
// Weight layout in wmem (dense, at woff): conv/pw k*Cout+oc ; dw ci*RS+k ;
// fc c*NC+oc. Params in pmem (at poff): conv/pw/dw per-channel mult/shift/bias ;
// fc per-class mult(=pmult)/bias(=pbias).
module dscnn_seq #(
    parameter int NL = 12, parameter int BUF = 8192,
    parameter int WMEM = 32768, parameter int PMEM = 1024,
    parameter int MAXM = 128, parameter int MAXK = 64, parameter int MAXP = 64,
    parameter int MAXC = 64, parameter int MAXRS = 16, parameter int MAXNC = 16
) (
    input  logic                       clk, rst,
    input  logic [3:0]                 nlayers,
    input  logic                       desc_wr_en,
    input  logic [4:0]                 desc_field,
    input  logic [3:0]                 desc_layer,
    input  logic signed [31:0]         desc_val,
    input  logic                       wmem_wr_en,
    input  logic [$clog2(WMEM)-1:0]    wmem_addr,
    input  logic signed [7:0]          wmem_data,
    input  logic                       pmem_wr_en,
    input  logic [1:0]                 pmem_sel,
    input  logic [$clog2(PMEM)-1:0]    pmem_addr,
    input  logic signed [31:0]         pmem_data,
    input  logic                       fm_wr_en,
    input  logic [$clog2(BUF)-1:0]     fm_wr_addr,
    input  logic signed [7:0]          fm_wr_data,
    input  logic                       start,
    output logic                       busy, done,
    output logic                       out_in_b,
    input  logic [$clog2(BUF)-1:0]     rd_addr,
    output logic signed [7:0]          rd_data,
    output logic [3:0]                 pred          // final predicted class (from FC)
);
    logic signed [31:0] desc [0:16][0:NL-1];
    logic signed [7:0]  wmem [0:WMEM-1];
    logic signed [31:0] pmult[0:PMEM-1], pbias[0:PMEM-1];
    logic [5:0]         pshift[0:PMEM-1];
    logic signed [7:0]  bufA [0:BUF-1];
    logic signed [7:0]  bufB [0:BUF-1];
    logic loading;

    always_ff @(posedge clk) begin
        if (loading & desc_wr_en) desc[desc_field][desc_layer] <= desc_val;
        if (loading & wmem_wr_en) wmem[wmem_addr] <= wmem_data;
        if (loading & pmem_wr_en) begin
            if      (pmem_sel==2'd0) pmult [pmem_addr] <= pmem_data;
            else if (pmem_sel==2'd1) pshift[pmem_addr] <= pmem_data[5:0];
            else                     pbias [pmem_addr] <= pmem_data;
        end
    end

    logic [3:0] li;
    logic [7:0] dH,dW,dC,dR,dS,dstr,dCo,dP,dQ; logic signed [7:0] dpt,dpl; logic drelu;
    logic [1:0] dtype;
    logic [31:0] dwoff, dpoff; logic signed [31:0] dpmult; logic [5:0] dpshift;
    always_comb begin
        dH=desc[0][li]; dW=desc[1][li]; dC=desc[2][li]; dR=desc[3][li]; dS=desc[4][li];
        dstr=desc[5][li]; dCo=desc[6][li]; dP=desc[7][li]; dQ=desc[8][li];
        dpt=desc[9][li]; dpl=desc[10][li]; drelu=desc[11][li][0];
        dwoff=desc[12][li]; dpoff=desc[13][li]; dtype=desc[14][li][1:0];
        dpmult=desc[15][li]; dpshift=desc[16][li][5:0];
    end
    logic [31:0] n_fm, n_out, kdim, rsdim, npar;
    assign kdim  = dR*dS*dC;
    assign rsdim = dR*dS;
    assign n_fm  = dH*dW*dC;
    assign n_out = (dtype==2'd1) ? dP*dQ*dC : dP*dQ*dCo;      // dw keeps C; conv/pw -> Cout
    assign npar  = (dtype==2'd0) ? dCo : (dtype==2'd1) ? dC : dCo; // #channels/classes

    // ---- state + counters ----
    typedef enum logic [3:0] {S_IDLE,S_LFM,S_LW,S_LDW,S_LFCW,S_LP,S_LFCP,S_RUN,S_WAIT,S_STORE,S_NEXT,S_DONE} state_t;
    state_t state;
    assign loading = (state == S_IDLE);
    logic [31:0] a; logic [7:0] kk, oo, pc; logic [1:0] ps;
    logic srcB; assign srcB = li[0];
    logic le_a, dw_a, fc_a;
    assign le_a = (dtype==2'd0); assign dw_a = (dtype==2'd1); assign fc_a = (dtype==2'd2);
    // bufA/bufB banked: one synchronous read port each, address muxed between the
    // feature-map load (index a during S_LFM) and the external result read (rd_addr).
    // fm_src/rd_data become registered (1-cycle) reads; the fm-load write control is
    // pipelined to match (below).
    logic signed [7:0] bufA_q, bufB_q;
    logic [$clog2(BUF)-1:0] buf_ra;
    assign buf_ra = (state==S_LFM) ? a[$clog2(BUF)-1:0] : rd_addr;
    always_ff @(posedge clk) begin bufA_q <= bufA[buf_ra]; bufB_q <= bufB[buf_ra]; end
    logic signed [7:0] fm_src; assign fm_src = srcB ? bufB_q : bufA_q;

    // ==== ONE shared 8x8 MAC array, time-multiplexed across the three engines ====
    // The engines are instantiated with EXT_GEMM=1, so none has a private array;
    // each exposes its gemm drive on xg_*. Exactly one engine is active per layer
    // (selected by dtype: le_a/dw_a/fc_a), so muxing the active engine's xg_* onto
    // a single gemm_top_bram collapses the former 3x64 = 192 DSPs down to 64.
    logic [$clog2(MAXM+1)-1:0] le_xg_m_dim,dw_xg_m_dim,fc_xg_m_dim;
    logic [$clog2(MAXK+1)-1:0] le_xg_k_dim,dw_xg_k_dim,fc_xg_k_dim;
    logic [$clog2(MAXP+1)-1:0] le_xg_p_dim,dw_xg_p_dim,fc_xg_p_dim;
    logic le_xg_wr_en,dw_xg_wr_en,fc_xg_wr_en;
    logic le_xg_wr_is_w,dw_xg_wr_is_w,fc_xg_wr_is_w;
    logic [$clog2(MAXM*MAXK)-1:0] le_xg_wr_addr,dw_xg_wr_addr,fc_xg_wr_addr;
    logic signed [7:0] le_xg_wr_data,dw_xg_wr_data,fc_xg_wr_data;
    logic le_xg_start,dw_xg_start,fc_xg_start;
    logic [$clog2(MAXM*MAXP)-1:0] le_xg_rd_addr,dw_xg_rd_addr,fc_xg_rd_addr;
    logic g_busy,g_done; logic signed [31:0] g_rd_data;

    logic [$clog2(MAXM+1)-1:0] g_m_dim; logic [$clog2(MAXK+1)-1:0] g_k_dim;
    logic [$clog2(MAXP+1)-1:0] g_p_dim;
    logic g_wr_en,g_wr_is_w; logic [$clog2(MAXM*MAXK)-1:0] g_wr_addr;
    logic signed [7:0] g_wr_data; logic g_start; logic [$clog2(MAXM*MAXP)-1:0] g_rd_addr;
    always_comb begin
        g_m_dim   = le_a ? le_xg_m_dim   : dw_a ? dw_xg_m_dim   : fc_xg_m_dim;
        g_k_dim   = le_a ? le_xg_k_dim   : dw_a ? dw_xg_k_dim   : fc_xg_k_dim;
        g_p_dim   = le_a ? le_xg_p_dim   : dw_a ? dw_xg_p_dim   : fc_xg_p_dim;
        g_wr_en   = le_a ? le_xg_wr_en   : dw_a ? dw_xg_wr_en   : fc_xg_wr_en;
        g_wr_is_w = le_a ? le_xg_wr_is_w : dw_a ? dw_xg_wr_is_w : fc_xg_wr_is_w;
        g_wr_addr = le_a ? le_xg_wr_addr : dw_a ? dw_xg_wr_addr : fc_xg_wr_addr;
        g_wr_data = le_a ? le_xg_wr_data : dw_a ? dw_xg_wr_data : fc_xg_wr_data;
        g_start   = le_a ? le_xg_start   : dw_a ? dw_xg_start   : fc_xg_start;
        g_rd_addr = le_a ? le_xg_rd_addr : dw_a ? dw_xg_rd_addr : fc_xg_rd_addr;
    end
    gemm_top_bram #(.N(8),.ACCW(32),.MAXM(MAXM),.MAXK(MAXK),.MAXP(MAXP)) u_gemm_shared (
        .clk(clk),.rst(rst),.m_dim(g_m_dim),.k_dim(g_k_dim),.p_dim(g_p_dim),
        .wr_en(g_wr_en),.wr_is_w(g_wr_is_w),.wr_addr(g_wr_addr),.wr_data(g_wr_data),
        .start(g_start),.busy(g_busy),.done(g_done),.rd_addr(g_rd_addr),.rd_data(g_rd_data));

    // ---- engines ----
    logic le_start,le_done,le_fm_we,le_w_we,le_pr_we;
    logic [$clog2(BUF)-1:0] le_fm_addr; logic signed [7:0] le_fm_data;
    logic [$clog2(MAXK*MAXP)-1:0] le_w_addr; logic signed [7:0] le_w_data;
    logic [1:0] le_pr_sel; logic [$clog2(MAXP)-1:0] le_pr_addr; logic signed [31:0] le_pr_data;
    logic [$clog2(MAXM*MAXP)-1:0] le_rd_addr; logic signed [7:0] le_rd_data;
    layer_engine #(.MAXFM(BUF),.MAXM(MAXM),.MAXK(MAXK),.MAXP(MAXP),.EXT_GEMM(1)) u_le (
        .clk(clk),.rst(rst),.H(dH),.W(dW),.C(dC),.R(dR),.S(dS),.stride(dstr),.Cout(dCo),
        .P(dP),.Q(dQ),.pad_top(dpt),.pad_left(dpl),.relu(drelu),
        .fm_wr_en(le_fm_we),.fm_wr_addr(le_fm_addr),.fm_wr_data(le_fm_data),
        .w_wr_en(le_w_we),.w_wr_addr(le_w_addr),.w_wr_data(le_w_data),
        .pr_wr_en(le_pr_we),.pr_sel(le_pr_sel),.pr_addr(le_pr_addr),.pr_data(le_pr_data),
        .start(le_start),.busy(),.done(le_done),.rd_addr(le_rd_addr),.rd_data(le_rd_data),
        .xg_m_dim(le_xg_m_dim),.xg_k_dim(le_xg_k_dim),.xg_p_dim(le_xg_p_dim),
        .xg_wr_en(le_xg_wr_en),.xg_wr_is_w(le_xg_wr_is_w),.xg_wr_addr(le_xg_wr_addr),
        .xg_wr_data(le_xg_wr_data),.xg_start(le_xg_start),.xg_rd_addr(le_xg_rd_addr),
        .xg_busy(g_busy),.xg_done(g_done),.xg_rd_data(g_rd_data));

    logic dw_start,dw_done,dw_fm_we,dw_w_we,dw_pr_we;
    logic [$clog2(BUF)-1:0] dw_fm_addr; logic signed [7:0] dw_fm_data;
    logic [$clog2(MAXC*MAXRS)-1:0] dw_w_addr; logic signed [7:0] dw_w_data;
    logic [1:0] dw_pr_sel; logic [$clog2(MAXC)-1:0] dw_pr_addr; logic signed [31:0] dw_pr_data;
    logic [$clog2(MAXM*MAXC)-1:0] dw_rd_addr; logic signed [7:0] dw_rd_data;
    dw_engine #(.MAXFM(BUF),.MAXM(MAXM),.MAXK(MAXK),.MAXP(MAXP),.MAXC(MAXC),.MAXRS(MAXRS),.EXT_GEMM(1)) u_dw (
        .clk(clk),.rst(rst),.H(dH),.W(dW),.C(dC),.R(dR),.S(dS),.stride(dstr),.P(dP),.Q(dQ),
        .pad_top(dpt),.pad_left(dpl),
        .fm_wr_en(dw_fm_we),.fm_wr_addr(dw_fm_addr),.fm_wr_data(dw_fm_data),
        .dw_wr_en(dw_w_we),.dw_wr_addr(dw_w_addr),.dw_wr_data(dw_w_data),
        .pr_wr_en(dw_pr_we),.pr_sel(dw_pr_sel),.pr_addr(dw_pr_addr),.pr_data(dw_pr_data),
        .start(dw_start),.busy(),.done(dw_done),.rd_addr(dw_rd_addr),.rd_data(dw_rd_data),
        .xg_m_dim(dw_xg_m_dim),.xg_k_dim(dw_xg_k_dim),.xg_p_dim(dw_xg_p_dim),
        .xg_wr_en(dw_xg_wr_en),.xg_wr_is_w(dw_xg_wr_is_w),.xg_wr_addr(dw_xg_wr_addr),
        .xg_wr_data(dw_xg_wr_data),.xg_start(dw_xg_start),.xg_rd_addr(dw_xg_rd_addr),
        .xg_busy(g_busy),.xg_done(g_done),.xg_rd_data(g_rd_data));

    logic fc_start,fc_done,fc_fm_we,fc_w_we,fc_p_we;
    logic [$clog2(BUF)-1:0] fc_fm_addr; logic signed [7:0] fc_fm_data;
    logic [$clog2(MAXK*MAXP)-1:0] fc_w_addr; logic signed [7:0] fc_w_data;
    logic fc_p_sel; logic [$clog2(MAXNC)-1:0] fc_p_addr; logic signed [31:0] fc_p_data;
    logic [3:0] fc_pred;
    fc_engine #(.MAXIN(BUF),.MAXC(MAXC),.MAXNC(MAXNC),.MAXM(MAXM),.MAXK(MAXK),.MAXP(MAXP),.EXT_GEMM(1)) u_fc (
        .clk(clk),.rst(rst),.M((dH*dW)>>0),.C(dC),.NC(dCo),
        .fm_wr_en(fc_fm_we),.fm_wr_addr(fc_fm_addr),.fm_wr_data(fc_fm_data),
        .w_wr_en(fc_w_we),.w_wr_addr(fc_w_addr),.w_wr_data(fc_w_data),
        .pool_mult(dpmult),.pool_shift(dpshift),
        .fcp_wr_en(fc_p_we),.fcp_sel(fc_p_sel),.fcp_addr(fc_p_addr),.fcp_data(fc_p_data),
        .start(fc_start),.busy(),.done(fc_done),.rd_addr('0),.vq_data(),.pred(fc_pred),
        .xg_m_dim(fc_xg_m_dim),.xg_k_dim(fc_xg_k_dim),.xg_p_dim(fc_xg_p_dim),
        .xg_wr_en(fc_xg_wr_en),.xg_wr_is_w(fc_xg_wr_is_w),.xg_wr_addr(fc_xg_wr_addr),
        .xg_wr_data(fc_xg_wr_data),.xg_start(fc_xg_start),.xg_rd_addr(fc_xg_rd_addr),
        .xg_busy(g_busy),.xg_done(g_done),.xg_rd_data(g_rd_data));

    // ---- shared load muxing ----
    logic signed [7:0] out_rd; assign out_rd = le_a ? le_rd_data : dw_rd_data;
    logic act_done; assign act_done = le_a ? le_done : dw_a ? dw_done : fc_done;

    // ---- wmem banked to BRAM: one synchronous read (muxed address, only one
    // weight-load state is ever active) + a 1-cycle pipeline on the engine
    // weight-write control so the write lands with the registered read data.
    // This removes the 32768-deep combinational read mux (the biggest LUT cost).
    logic signed [7:0]           wmem_q;
    logic [$clog2(WMEM)-1:0]     wmem_ra;
    always_comb wmem_ra = (state==S_LDW) ? (dwoff + oo*int'(rsdim) + kk)
                        : (state==S_LFCW)? (dwoff + oo*int'(dCo)   + kk)
                        :                  (dwoff + kk*int'(dCo)   + oo);  // S_LW
    logic le_w_we_q, dw_w_we_q, fc_w_we_q;
    logic [$clog2(MAXK*MAXP)-1:0]  le_w_addr_q, fc_w_addr_q;
    logic [$clog2(MAXC*MAXRS)-1:0] dw_w_addr_q;
    // fm-load write control, pipelined 1 cycle to match the registered bufA/bufB read
    logic le_fm_we_q, dw_fm_we_q, fc_fm_we_q;
    logic [$clog2(BUF)-1:0] fm_addr_q;
    always_ff @(posedge clk) begin
        wmem_q      <= wmem[wmem_ra];
        le_w_we_q   <= (state==S_LW);   le_w_addr_q <= kk*MAXP  + oo;
        dw_w_we_q   <= (state==S_LDW);  dw_w_addr_q <= oo*MAXRS + kk;
        fc_w_we_q   <= (state==S_LFCW); fc_w_addr_q <= oo*MAXP  + kk;
        le_fm_we_q  <= le_a & (state==S_LFM);
        dw_fm_we_q  <= dw_a & (state==S_LFM);
        fc_fm_we_q  <= fc_a & (state==S_LFM);
        fm_addr_q   <= a[$clog2(BUF)-1:0];
    end
    // bufA/bufB unified single write port each: load into bufA while idle, and
    // store the active engine's output into the destination buffer during S_STORE.
    // One write + one registered read port per buffer => BRAM, not LUT-RAM.
    always_ff @(posedge clk) begin
        if ((loading & fm_wr_en) | (state==S_STORE & srcB))      // srcB -> dest is bufA
            bufA[loading ? fm_wr_addr : a[$clog2(BUF)-1:0]] <= loading ? fm_wr_data : out_rd;
        if (state==S_STORE & ~srcB)                              // ~srcB -> dest is bufB
            bufB[a[$clog2(BUF)-1:0]] <= out_rd;
    end

    always_comb begin
        // feature-map load: pipelined write control + registered read data (fm_src)
        le_fm_we = le_fm_we_q; le_fm_addr = fm_addr_q; le_fm_data = fm_src;
        dw_fm_we = dw_fm_we_q; dw_fm_addr = fm_addr_q; dw_fm_data = fm_src;
        fc_fm_we = fc_fm_we_q; fc_fm_addr = fm_addr_q; fc_fm_data = fm_src;
        // read addr for store (le/dw share linear index a)
        le_rd_addr = a[$clog2(MAXM*MAXP)-1:0];
        dw_rd_addr = a[$clog2(MAXM*MAXC)-1:0];
        // weights: pipelined write control (1-cycle delay) + registered wmem read.
        le_w_we = le_w_we_q; le_w_addr = le_w_addr_q; le_w_data = wmem_q;  // conv/pw
        dw_w_we = dw_w_we_q; dw_w_addr = dw_w_addr_q; dw_w_data = wmem_q;  // depthwise
        fc_w_we = fc_w_we_q; fc_w_addr = fc_w_addr_q; fc_w_data = wmem_q;  // fc
        // per-channel requant params (conv/pw -> le, dw -> dw)
        le_pr_we = le_a & (state==S_LP); le_pr_sel = ps; le_pr_addr = pc[$clog2(MAXP)-1:0];
        le_pr_data = (ps==2'd0)? pmult[dpoff+pc] : (ps==2'd1)? {26'b0,pshift[dpoff+pc]} : pbias[dpoff+pc];
        dw_pr_we = dw_a & (state==S_LP); dw_pr_sel = ps; dw_pr_addr = pc[$clog2(MAXC)-1:0];
        dw_pr_data = (ps==2'd0)? pmult[dpoff+pc] : (ps==2'd1)? {26'b0,pshift[dpoff+pc]} : pbias[dpoff+pc];
        // fc per-class params (sel 0=mult from pmult, 1=bias from pbias)
        fc_p_we = (state==S_LFCP); fc_p_sel = ps[0]; fc_p_addr = pc[$clog2(MAXNC)-1:0];
        fc_p_data = (ps==2'd0)? pmult[dpoff+pc] : pbias[dpoff+pc];
    end

    always_ff @(posedge clk) begin
        if (rst) begin
            state<=S_IDLE; li<='0; a<='0; kk<='0; oo<='0; pc<='0; ps<='0;
            le_start<=1'b0; dw_start<=1'b0; fc_start<=1'b0; busy<=1'b0; done<=1'b0; out_in_b<=1'b0; pred<='0;
        end else begin
            le_start<=1'b0; dw_start<=1'b0; fc_start<=1'b0;
            case (state)
                S_IDLE: begin done<=1'b0; if (start) begin busy<=1'b1; li<='0; a<='0; state<=S_LFM; end end
                S_LFM: if (a == n_fm-1) begin a<='0; kk<='0; oo<='0;
                           if (le_a) state<=S_LW; else if (dw_a) state<=S_LDW; else state<=S_LFCW;
                       end else a<=a+32'd1;
                S_LW: if (oo == dCo-1) begin oo<='0;                       // conv/pw: k outer, oc inner
                          if (kk == kdim-1) begin kk<='0; pc<='0; ps<='0; state<=S_LP; end else kk<=kk+8'd1;
                      end else oo<=oo+8'd1;
                S_LDW: if (kk == rsdim-1) begin kk<='0;                    // dw: ci outer(oo), k inner(kk)
                           if (oo == dC-1) begin oo<='0; pc<='0; ps<='0; state<=S_LP; end else oo<=oo+8'd1;
                       end else kk<=kk+8'd1;
                S_LFCW: if (kk == dCo-1) begin kk<='0;                     // fc: c outer(oo), oc inner(kk)
                            if (oo == dC-1) begin oo<='0; pc<='0; ps<='0; state<=S_LFCP; end else oo<=oo+8'd1;
                        end else kk<=kk+8'd1;
                S_LP: if (ps == 2'd2) begin ps<='0;                        // 3 params/channel
                          if (pc == npar-1) begin pc<='0; state<=S_RUN; end else pc<=pc+8'd1;
                      end else ps<=ps+2'd1;
                S_LFCP: if (ps == 2'd1) begin ps<='0;                      // 2 params/class (mult,bias)
                            if (pc == npar-1) begin pc<='0; state<=S_RUN; end else pc<=pc+8'd1;
                        end else ps<=ps+2'd1;
                S_RUN: begin
                    if (le_a) le_start<=1'b1; else if (dw_a) dw_start<=1'b1; else fc_start<=1'b1;
                    state<=S_WAIT;
                end
                S_WAIT: if (act_done) begin a<='0;
                            if (fc_a) begin pred<=fc_pred; state<=S_NEXT; end else state<=S_STORE;
                        end
                S_STORE: begin                       // buf writes handled by the unified port above
                    if (a == n_out-1) begin a<='0; state<=S_NEXT; end else a<=a+32'd1;
                end
                S_NEXT: if (li == nlayers-1) begin out_in_b <= ~li[0]; state<=S_DONE; end
                        else begin li<=li+4'd1; a<='0; state<=S_LFM; end
                S_DONE: begin busy<=1'b0; done<=1'b1; if (!start) state<=S_IDLE; end
                default: state<=S_IDLE;
            endcase
        end
    end

    // registered (1-cycle) external read of the final feature-map buffer
    assign rd_data = out_in_b ? bufB_q : bufA_q;
endmodule
