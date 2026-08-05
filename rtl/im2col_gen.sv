// im2col_gen.sv
// Phase 3 of the hardware sequencer: the im2col (convolution-lowering) address
// generator. It reads a feature map and produces the lowered "columns" matrix
//   cols[m=p*Q+q][k=(r*S+s)*C+c] = xpad[p*stride+r][q*stride+s][c]
// that the tiling GEMM engine multiplies by the flattened weights -- exactly what
// accel_sim._im2col builds in software (matched flattening: r-major, then s, then
// c). TF "SAME" padding is done implicitly: source positions outside [0,H)/[0,W)
// read as 0 (pad_top/pad_left are the asymmetric SAME offsets). Standard/pointwise
// convs use C = Cin; depthwise runs one channel at a time with C = 1.
// One cols element per cycle; buffers are behavioural (as elsewhere in this repo).
module im2col_gen #(
    parameter int MAXFM = 8192, parameter int MAXM = 128, parameter int MAXK = 64
) (
    input  logic                          clk, rst,
    // ---- convolution geometry (runtime) ----
    // C = channels EMITTED into cols (Cin for std/pw; 1 for a single depthwise
    // channel). Cstride/cbase locate the source: src channel = cbase + c, and the
    // source map has Cstride channels per pixel. For std/pw set Cstride=C,cbase=0.
    input  logic [7:0]                    H, W, C, R, S, stride, P, Q,
    input  logic [7:0]                    Cstride, cbase,
    input  logic signed [7:0]             pad_top, pad_left,
    // ---- feature-map load: fmap[(row*W + col)*C + c] ----
    input  logic                          wr_en,
    input  logic [$clog2(MAXFM)-1:0]      wr_addr,
    input  logic signed [7:0]             wr_data,
    // ---- control ----
    input  logic                          start,
    output logic                          busy, done,
    // ---- cols read: cols[m*MAXK + k] ----
    input  logic [$clog2(MAXM*MAXK)-1:0]  rd_addr,
    output logic signed [7:0]             rd_data
);
    logic signed [7:0] fmap [0:MAXFM-1];
    logic signed [7:0] cols [0:MAXM*MAXK-1];

    always_ff @(posedge clk) if (wr_en) fmap[wr_addr] <= wr_data;
    assign rd_data = cols[rd_addr];

    // ---- iteration counters: p,q (output pos) x r,s,c (kernel tap, channel) ----
    logic [7:0] p, q, r, s, c;

    // current element's geometry (signed math so padding can go negative)
    int  sr, sc, coladdr, srcaddr;
    logic inb;
    logic [$clog2(MAXFM)-1:0] safe_src;
    always_comb begin
        sr      = int'(p)*int'(stride) + int'(r) - int'(pad_top);
        sc      = int'(q)*int'(stride) + int'(s) - int'(pad_left);
        inb     = (sr >= 0) && (sr < int'(H)) && (sc >= 0) && (sc < int'(W));
        srcaddr = (sr*int'(W) + sc)*int'(Cstride) + int'(cbase) + int'(c);
        coladdr = (int'(p)*int'(Q) + int'(q))*MAXK + (int'(r)*int'(S) + int'(s))*int'(C) + int'(c);
        safe_src = inb ? srcaddr[$clog2(MAXFM)-1:0] : '0;
    end

    typedef enum logic [1:0] {S_IDLE, S_GEN, S_DONE} state_t;
    state_t state;

    always_ff @(posedge clk) begin
        if (rst) begin
            state <= S_IDLE; p<='0; q<='0; r<='0; s<='0; c<='0;
            busy <= 1'b0; done <= 1'b0;
        end else case (state)
            S_IDLE: begin
                done <= 1'b0;
                if (start) begin
                    busy <= 1'b1; p<='0; q<='0; r<='0; s<='0; c<='0; state <= S_GEN;
                end
            end
            S_GEN: begin
                cols[coladdr] <= inb ? fmap[safe_src] : 8'sd0;   // emit one element
                // odometer increment over c -> s -> r -> q -> p
                if (c != C-1) c <= c + 8'd1;
                else begin c <= '0;
                    if (s != S-1) s <= s + 8'd1;
                    else begin s <= '0;
                        if (r != R-1) r <= r + 8'd1;
                        else begin r <= '0;
                            if (q != Q-1) q <= q + 8'd1;
                            else begin q <= '0;
                                if (p != P-1) p <= p + 8'd1;
                                else state <= S_DONE;             // last element emitted
                            end
                        end
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
endmodule
