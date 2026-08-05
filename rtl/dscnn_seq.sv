// dscnn_seq.sv
// Multi-layer sequencer (Phase 4b, mechanism): walks a per-layer descriptor table
// and runs each convolution/pointwise layer on layer_engine, chaining them in
// hardware with two ping-pong feature-map buffers. Because a layer's output map
// out[m*Cout+oc] is addressed identically to the next layer's input map
// [(p*Q+q)*Cout+oc], chaining is a plain linear copy -- no re-addressing.
//
// The host loads (while idle): the descriptor table (desc_*), the concatenated
// int8 weights of every layer (wmem_*), the per-output-channel requant params
// (pmem_*), and the input feature map into buffer A (fm_*). Pulse start; when done,
// the final layer's output map is in the buffer selected by `out_in_b`, readable
// at rd_addr. This version handles standard/pointwise layers (layer_engine);
// depthwise/FC dispatch is the next step.
module dscnn_seq #(
    parameter int NL = 8, parameter int BUF = 8192,
    parameter int WMEM = 32768, parameter int PMEM = 1024,
    parameter int MAXM = 128, parameter int MAXK = 64, parameter int MAXP = 64
) (
    input  logic                       clk, rst,
    input  logic [3:0]                 nlayers,
    // descriptor write: desc[field][layer] = val   (host, while idle)
    input  logic                       desc_wr_en,
    input  logic [3:0]                 desc_field,   // 0H 1W 2C 3R 4S 5stride 6Cout 7P 8Q 9pt 10pl 11relu 12woff 13poff
    input  logic [3:0]                 desc_layer,
    input  logic signed [31:0]         desc_val,
    // concatenated weights (dense k*Cout+oc per layer, at woff)
    input  logic                       wmem_wr_en,
    input  logic [$clog2(WMEM)-1:0]    wmem_addr,
    input  logic signed [7:0]          wmem_data,
    // per-output-channel requant params (global index = poff+oc)
    input  logic                       pmem_wr_en,
    input  logic [1:0]                 pmem_sel,     // 0 mult 1 shift 2 bias
    input  logic [$clog2(PMEM)-1:0]    pmem_addr,
    input  logic signed [31:0]         pmem_data,
    // input feature map -> buffer A
    input  logic                       fm_wr_en,
    input  logic [$clog2(BUF)-1:0]     fm_wr_addr,
    input  logic signed [7:0]          fm_wr_data,
    // control + final-output read
    input  logic                       start,
    output logic                       busy, done,
    output logic                       out_in_b,     // final map is in buffer B?
    input  logic [$clog2(BUF)-1:0]     rd_addr,
    output logic signed [7:0]          rd_data
);
    // ---- descriptor table + weight/param/feature memories ----
    logic signed [31:0] desc [0:13][0:NL-1];
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
        if (loading & fm_wr_en) bufA[fm_wr_addr] <= fm_wr_data;
    end

    // ---- current layer descriptor (unpacked) ----
    logic [3:0] li;
    logic [7:0] dH,dW,dC,dR,dS,dstr,dCo,dP,dQ; logic signed [7:0] dpt,dpl; logic drelu;
    logic [31:0] dwoff, dpoff;
    always_comb begin
        dH=desc[0][li]; dW=desc[1][li]; dC=desc[2][li]; dR=desc[3][li]; dS=desc[4][li];
        dstr=desc[5][li]; dCo=desc[6][li]; dP=desc[7][li]; dQ=desc[8][li];
        dpt=desc[9][li]; dpl=desc[10][li]; drelu=desc[11][li][0];
        dwoff=desc[12][li]; dpoff=desc[13][li];
    end
    // element counts for this layer (widened; see the ca*Cout truncation lesson)
    logic [31:0] n_fm, n_out, kdim;
    assign kdim  = dR*dS*dC;                       // im2col inner dim
    assign n_fm  = dH*dW*dC;                        // input map size
    assign n_out = dP*dQ*dCo;                       // output map size

    // ---- one layer_engine, driven per layer ----
    logic le_start, le_busy, le_done;
    logic le_fm_we; logic [$clog2(BUF)-1:0] le_fm_addr; logic signed [7:0] le_fm_data;
    logic le_w_we;  logic [$clog2(MAXK*MAXP)-1:0] le_w_addr; logic signed [7:0] le_w_data;
    logic le_pr_we; logic [1:0] le_pr_sel; logic [$clog2(MAXP)-1:0] le_pr_addr; logic signed [31:0] le_pr_data;
    logic [$clog2(MAXM*MAXP)-1:0] le_rd_addr; logic signed [7:0] le_rd_data;
    layer_engine #(.MAXFM(BUF),.MAXM(MAXM),.MAXK(MAXK),.MAXP(MAXP)) u_le (
        .clk(clk),.rst(rst),.H(dH),.W(dW),.C(dC),.R(dR),.S(dS),.stride(dstr),.Cout(dCo),
        .P(dP),.Q(dQ),.pad_top(dpt),.pad_left(dpl),.relu(drelu),
        .fm_wr_en(le_fm_we),.fm_wr_addr(le_fm_addr),.fm_wr_data(le_fm_data),
        .w_wr_en(le_w_we),.w_wr_addr(le_w_addr),.w_wr_data(le_w_data),
        .pr_wr_en(le_pr_we),.pr_sel(le_pr_sel),.pr_addr(le_pr_addr),.pr_data(le_pr_data),
        .start(le_start),.busy(le_busy),.done(le_done),.rd_addr(le_rd_addr),.rd_data(le_rd_data));

    // ---- sequencer FSM ----
    typedef enum logic [3:0] {S_IDLE,S_LFM,S_LW,S_LP,S_RUN,S_WAIT,S_STORE,S_NEXT,S_DONE} state_t;
    state_t state;
    assign loading = (state == S_IDLE);
    logic [31:0] a;            // primary copy counter
    logic [7:0]  kk, oo;       // weight (k,oc) counters
    logic [7:0]  pc;           // param channel
    logic [1:0]  ps;           // param sub-field
    logic        srcB;         // current source buffer is B?

    assign srcB   = li[0];                     // layer 0 reads A, writes B; layer 1 reads B; ...

    logic signed [7:0] fm_src;  assign fm_src = srcB ? bufB[a[$clog2(BUF)-1:0]] : bufA[a[$clog2(BUF)-1:0]];

    // layer_engine load muxing
    always_comb begin
        le_fm_we=1'b0; le_fm_addr='0; le_fm_data='0;
        le_w_we=1'b0;  le_w_addr='0;  le_w_data='0;
        le_pr_we=1'b0; le_pr_sel='0;  le_pr_addr='0; le_pr_data='0;
        le_rd_addr = a[$clog2(MAXM*MAXP)-1:0];
        case (state)
            S_LFM: begin le_fm_we=1'b1; le_fm_addr=a[$clog2(BUF)-1:0]; le_fm_data=fm_src; end
            S_LW:  begin le_w_we=1'b1;  le_w_addr=kk*MAXP+oo; le_w_data=wmem[dwoff + kk*int'(dCo) + oo]; end
            S_LP:  begin le_pr_we=1'b1; le_pr_sel=ps; le_pr_addr=pc[$clog2(MAXP)-1:0];
                         le_pr_data = (ps==2'd0)? pmult[dpoff+pc] : (ps==2'd1)? {26'b0,pshift[dpoff+pc]} : pbias[dpoff+pc]; end
            default: ;
        endcase
    end

    always_ff @(posedge clk) begin
        if (rst) begin
            state<=S_IDLE; li<='0; a<='0; kk<='0; oo<='0; pc<='0; ps<='0;
            le_start<=1'b0; busy<=1'b0; done<=1'b0; out_in_b<=1'b0;
        end else begin
            le_start<=1'b0;
            case (state)
                S_IDLE: begin done<=1'b0;
                    if (start) begin busy<=1'b1; li<='0; a<='0; state<=S_LFM; end
                end
                S_LFM: begin                                   // copy src buffer -> le.fm
                    if (a == n_fm-1) begin a<='0; kk<='0; oo<='0; state<=S_LW; end
                    else a<=a+32'd1;
                end
                S_LW: begin                                    // copy weights -> le (k*MAXP+oc)
                    if (oo == dCo-1) begin oo<='0;
                        if (kk == kdim-1) begin kk<='0; pc<='0; ps<='0; state<=S_LP; end
                        else kk<=kk+8'd1;
                    end else oo<=oo+8'd1;
                end
                S_LP: begin                                    // per-channel requant params
                    if (ps == 2'd2) begin ps<='0;
                        if (pc == dCo-1) begin pc<='0; state<=S_RUN; end
                        else pc<=pc+8'd1;
                    end else ps<=ps+2'd1;
                end
                S_RUN:  begin le_start<=1'b1; state<=S_WAIT; end
                S_WAIT: if (le_done) begin a<='0; state<=S_STORE; end
                S_STORE: begin                                 // le output -> dst buffer (linear)
                    if (srcB) bufA[a[$clog2(BUF)-1:0]] <= le_rd_data;
                    else      bufB[a[$clog2(BUF)-1:0]] <= le_rd_data;
                    if (a == n_out-1) begin a<='0; state<=S_NEXT; end
                    else a<=a+32'd1;
                end
                S_NEXT: begin
                    if (li == nlayers-1) begin
                        out_in_b <= ~li[0];       // dst of last layer: A if li odd, B if li even
                        state<=S_DONE;
                    end else begin li<=li+4'd1; a<='0; state<=S_LFM; end
                end
                S_DONE: begin busy<=1'b0; done<=1'b1; if (!start) state<=S_IDLE; end
                default: state<=S_IDLE;
            endcase
        end
    end

    assign rd_data = out_in_b ? bufB[rd_addr] : bufA[rd_addr];
endmodule
