// tb_layer.sv
// Self-checking testbench for the single-layer engine (Phase 4a): runs a full
// conv/pointwise layer (im2col -> tiled GEMM -> per-channel requant) in hardware
// and checks the int8 output feature map against a golden that composes the same
// im2col + matmul + fixed-point requant here.
`timescale 1ns/1ps
module tb_layer;
    localparam int MAXFM = 8192, MAXM = 128, MAXK = 64, MAXP = 64;

    logic clk = 0, rst = 1;
    logic [7:0]  H,W,C,R,S,stride,Cout,P,Q;
    logic signed [7:0] pad_top, pad_left;
    logic relu;
    logic fm_wr_en; logic [$clog2(MAXFM)-1:0] fm_wr_addr; logic signed [7:0] fm_wr_data;
    logic w_wr_en;  logic [$clog2(MAXK*MAXP)-1:0] w_wr_addr; logic signed [7:0] w_wr_data;
    logic pr_wr_en; logic [1:0] pr_sel; logic [$clog2(MAXP)-1:0] pr_addr; logic signed [31:0] pr_data;
    logic start, busy, done;
    logic [$clog2(MAXM*MAXP)-1:0] rd_addr; logic signed [7:0] rd_data;

    layer_engine #(.MAXFM(MAXFM),.MAXM(MAXM),.MAXK(MAXK),.MAXP(MAXP)) dut (
        .clk(clk),.rst(rst),.H(H),.W(W),.C(C),.R(R),.S(S),.stride(stride),.Cout(Cout),
        .P(P),.Q(Q),.pad_top(pad_top),.pad_left(pad_left),.relu(relu),
        .fm_wr_en(fm_wr_en),.fm_wr_addr(fm_wr_addr),.fm_wr_data(fm_wr_data),
        .w_wr_en(w_wr_en),.w_wr_addr(w_wr_addr),.w_wr_data(w_wr_data),
        .pr_wr_en(pr_wr_en),.pr_sel(pr_sel),.pr_addr(pr_addr),.pr_data(pr_data),
        .start(start),.busy(busy),.done(done),.rd_addr(rd_addr),.rd_data(rd_data));

    always #5 clk = ~clk;

    logic signed [7:0] fm  [0:MAXFM-1];
    logic signed [7:0] wm  [0:MAXK-1][0:MAXP-1];
    logic signed [31:0] mlt[0:MAXP-1], bia[0:MAXP-1];
    logic [5:0]        shf[0:MAXP-1];
    int Oref [0:MAXM-1][0:MAXP-1];
    int errors = 0, tests = 0;

    function automatic logic signed [7:0] rq(input longint acc, m, b, input logic [5:0] sh, input logic rl);
        longint x, half, r;
        x = acc + b; half = (sh!=0) ? (longint'(1) <<< (sh-1)) : 0;
        r = (x*m + half) >>> sh;
        if (rl && r < 0) r = 0;
        if (r > 127) return 8'sd127; else if (r < -127) return -8'sd127; else return r[7:0];
    endfunction

    task automatic run_layer(input int iH,iW,iC,iR,iS,istr,iCo,iP,iQ,ipt,ipl, input logic rl);
        tests++;
        H=iH;W=iW;C=iC;R=iR;S=iS;stride=istr;Cout=iCo;P=iP;Q=iQ;pad_top=ipt;pad_left=ipl;relu=rl;

        // random operands + per-channel requant params
        for (int a=0;a<iH*iW*iC;a++) begin logic signed [7:0] t; t=$random; fm[a]=t; end
        for (int k=0;k<iR*iS*iC;k++) for (int o=0;o<iCo;o++) begin logic signed[7:0] t; t=$random; wm[k][o]=t; end
        for (int o=0;o<iCo;o++) begin
            mlt[o] = (1<<20) + (o*131);            // ~2^20 scale, slight per-channel spread
            shf[o] = 6'd33;                          // /2^33 * 2^20 = /2^13
            bia[o] = ($random % 4096);
        end

        // golden: im2col + matmul + requant
        for (int p=0;p<iP;p++) for (int q=0;q<iQ;q++) for (int o=0;o<iCo;o++) begin
            longint acc; acc = 0;
            for (int r=0;r<iR;r++) for (int s=0;s<iS;s++) for (int c=0;c<iC;c++) begin
                int sr,sc,kk; logic signed [7:0] a;
                sr=p*istr+r-ipt; sc=q*istr+s-ipl; kk=(r*iS+s)*iC+c;
                a = (sr>=0&&sr<iH&&sc>=0&&sc<iW) ? fm[(sr*iW+sc)*iC+c] : 8'sd0;
                acc += a * wm[kk][o];
            end
            Oref[p*iQ+q][o] = rq(acc, mlt[o], bia[o], shf[o], rl);
        end

        // ---- load feature map, weights, params ----
        @(negedge clk);
        for (int a=0;a<iH*iW*iC;a++) begin fm_wr_en=1; fm_wr_addr=a; fm_wr_data=fm[a]; @(negedge clk); end
        fm_wr_en=0;
        for (int k=0;k<iR*iS*iC;k++) for (int o=0;o<iCo;o++) begin
            w_wr_en=1; w_wr_addr=k*MAXP+o; w_wr_data=wm[k][o]; @(negedge clk); end
        w_wr_en=0;
        for (int o=0;o<iCo;o++) begin
            pr_wr_en=1; pr_sel=0; pr_addr=o; pr_data=mlt[o]; @(negedge clk);
            pr_sel=1; pr_data=shf[o]; @(negedge clk);
            pr_sel=2; pr_data=bia[o]; @(negedge clk);
        end
        pr_wr_en=0;

        // run + check
        start=1; @(negedge clk); start=0;
        wait (done); @(negedge clk);
        for (int m=0;m<iP*iQ;m++) for (int o=0;o<iCo;o++) begin
            rd_addr = m*iCo + o; #1;
            if (rd_data !== Oref[m][o]) begin
                errors++;
                if (errors<=12) $display("MISMATCH out[m=%0d][oc=%0d]: got %0d exp %0d", m,o,rd_data,Oref[m][o]);
            end
        end
        $display("  layer H%0d W%0d C%0d %0dx%0d s%0d -> Cout%0d (%0dx%0d) relu%0b : %0d elems %s",
                 iH,iW,iC,iR,iS,istr,iCo,iP,iQ,rl,iP*iQ*iCo,(errors==0)?"":"<-- FAIL");
    endtask

    initial begin
        fm_wr_en=0; w_wr_en=0; pr_wr_en=0; start=0; rd_addr=0;
        @(negedge clk); rst=0; @(negedge clk);

        run_layer(25,5, 8,  1,1, 1, 8,  25,5, 0,0, 1);  // pointwise 1x1, 8->8
        run_layer(13,6, 1,  3,2, 2, 8,   7,3, 1,0, 1);  // standard conv, SAME stride 2
        run_layer(25,5, 64, 1,1, 1, 64, 25,5, 0,0, 1);  // full-width pointwise 64->64

        if (errors==0) $display("PASS: %0d layer-engine tests match golden (im2col->GEMM->requant in HW)", tests);
        else           $display("FAIL: %0d mismatched elements across %0d tests", errors, tests);
        $finish;
    end
endmodule
