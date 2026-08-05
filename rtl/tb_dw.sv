// tb_dw.sv
// Self-checking testbench for the depthwise engine. Runs a full depthwise 3x3
// layer (per-channel conv + per-channel requant) in hardware and checks the int8
// output map against a golden that composes the same per-channel im2col + matmul
// + fixed-point requant here.
`timescale 1ns/1ps
module tb_dw;
    localparam int MAXFM=8192, MAXM=128, MAXK=64, MAXP=64, MAXC=64, MAXRS=16;

    logic clk=0, rst=1;
    logic [7:0] H,W,C,R,S,stride,P,Q; logic signed [7:0] pad_top,pad_left;
    logic fm_wr_en; logic [$clog2(MAXFM)-1:0] fm_wr_addr; logic signed [7:0] fm_wr_data;
    logic dw_wr_en; logic [$clog2(MAXC*MAXRS)-1:0] dw_wr_addr; logic signed [7:0] dw_wr_data;
    logic pr_wr_en; logic [1:0] pr_sel; logic [$clog2(MAXC)-1:0] pr_addr; logic signed [31:0] pr_data;
    logic start, busy, done;
    logic [$clog2(MAXM*MAXC)-1:0] rd_addr; logic signed [7:0] rd_data;

    dw_engine #(.MAXFM(MAXFM),.MAXM(MAXM),.MAXK(MAXK),.MAXP(MAXP),.MAXC(MAXC),.MAXRS(MAXRS)) dut (
        .clk(clk),.rst(rst),.H(H),.W(W),.C(C),.R(R),.S(S),.stride(stride),.P(P),.Q(Q),
        .pad_top(pad_top),.pad_left(pad_left),
        .fm_wr_en(fm_wr_en),.fm_wr_addr(fm_wr_addr),.fm_wr_data(fm_wr_data),
        .dw_wr_en(dw_wr_en),.dw_wr_addr(dw_wr_addr),.dw_wr_data(dw_wr_data),
        .pr_wr_en(pr_wr_en),.pr_sel(pr_sel),.pr_addr(pr_addr),.pr_data(pr_data),
        .start(start),.busy(busy),.done(done),.rd_addr(rd_addr),.rd_data(rd_data));

    always #5 clk = ~clk;

    logic signed [7:0] fm [0:MAXFM-1];
    logic signed [7:0] dw [0:MAXC-1][0:MAXRS-1];
    logic signed [31:0] mlt[0:MAXC-1], bia[0:MAXC-1]; logic [5:0] shf[0:MAXC-1];
    int Oref [0:MAXM-1][0:MAXC-1];
    int errors=0, tests=0;

    function automatic logic signed [7:0] rq(input longint acc,m,b,input logic[5:0] sh);
        longint x,half,r; x=acc+b; half=(sh!=0)?(longint'(1)<<<(sh-1)):0;
        r=(x*m+half)>>>sh; if (r<0) r=0;
        if (r>127) return 8'sd127; else return r[7:0];
    endfunction

    task automatic run_dw(input int iH,iW,iC,iR,iS,istr,iP,iQ,ipt,ipl);
        tests++;
        H=iH;W=iW;C=iC;R=iR;S=iS;stride=istr;P=iP;Q=iQ;pad_top=ipt;pad_left=ipl;
        for (int a=0;a<iH*iW*iC;a++) begin logic signed[7:0] t; t=$random; fm[a]=t; end
        for (int c=0;c<iC;c++) for (int k=0;k<iR*iS;k++) begin logic signed[7:0] t; t=$random; dw[c][k]=t; end
        for (int c=0;c<iC;c++) begin mlt[c]=(1<<20)+(c*97); shf[c]=6'd33; bia[c]=($random%2048); end

        // golden: per channel per position
        for (int c=0;c<iC;c++) for (int p=0;p<iP;p++) for (int q=0;q<iQ;q++) begin
            longint acc; acc=0;
            for (int r=0;r<iR;r++) for (int s=0;s<iS;s++) begin
                int sr,sc; logic signed[7:0] a;
                sr=p*istr+r-ipt; sc=q*istr+s-ipl;
                a=(sr>=0&&sr<iH&&sc>=0&&sc<iW)?fm[(sr*iW+sc)*iC+c]:8'sd0;
                acc += a*dw[c][r*iS+s];
            end
            Oref[p*iQ+q][c] = rq(acc, mlt[c], bia[c], shf[c]);
        end

        // load fmap, dw weights, params
        @(negedge clk);
        for (int a=0;a<iH*iW*iC;a++) begin fm_wr_en=1; fm_wr_addr=a; fm_wr_data=fm[a]; @(negedge clk); end
        fm_wr_en=0;
        for (int c=0;c<iC;c++) for (int k=0;k<iR*iS;k++) begin
            dw_wr_en=1; dw_wr_addr=c*MAXRS+k; dw_wr_data=dw[c][k]; @(negedge clk); end
        dw_wr_en=0;
        for (int c=0;c<iC;c++) begin
            pr_wr_en=1; pr_sel=0; pr_addr=c; pr_data=mlt[c]; @(negedge clk);
            pr_sel=1; pr_data=shf[c]; @(negedge clk);
            pr_sel=2; pr_data=bia[c]; @(negedge clk); end
        pr_wr_en=0;

        start=1; @(negedge clk); start=0; wait(done); @(negedge clk);
        for (int m=0;m<iP*iQ;m++) for (int c=0;c<iC;c++) begin
            rd_addr=m*iC+c; #1;
            if (rd_data!==Oref[m][c]) begin errors++;
                if (errors<=12) $display("MISMATCH out[m=%0d][ch=%0d]: got %0d exp %0d",m,c,rd_data,Oref[m][c]); end
        end
        $display("  depthwise H%0d W%0d C%0d %0dx%0d s%0d (%0dx%0d) : %0d elems %s",
                 iH,iW,iC,iR,iS,istr,iP,iQ,iP*iQ*iC,(errors==0)?"":"<-- FAIL");
    endtask

    initial begin
        fm_wr_en=0; dw_wr_en=0; pr_wr_en=0; start=0; rd_addr=0;
        @(negedge clk); rst=0; @(negedge clk);

        run_dw(25,5, 8,  3,3, 1, 25,5, 1,1);   // depthwise 3x3, 8 channels, SAME
        run_dw(25,5, 64, 3,3, 1, 25,5, 1,1);   // full-width depthwise 64 channels

        if (errors==0) $display("PASS: %0d depthwise-engine tests match golden (per-channel conv+requant in HW)", tests);
        else           $display("FAIL: %0d mismatched elements across %0d tests", errors, tests);
        $finish;
    end
endmodule
