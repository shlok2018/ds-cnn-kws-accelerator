// tb_im2col.sv
// Self-checking testbench for the im2col address generator (Phase 3). Loads a
// random feature map, runs im2col_gen at the real DS-CNN conv/dw/pw geometries
// (plus an odd strided case), and checks every cols element against a direct
// nested-loop im2col reference computed here.
`timescale 1ns/1ps
module tb_im2col;
    localparam int MAXFM = 8192, MAXM = 128, MAXK = 64;

    logic clk = 0, rst = 1;
    logic [7:0]                    H, W, C, R, S, stride, P, Q;
    logic signed [7:0]             pad_top, pad_left;
    logic                          wr_en;
    logic [$clog2(MAXFM)-1:0]      wr_addr;
    logic signed [7:0]             wr_data;
    logic                          start, busy, done;
    logic [$clog2(MAXM*MAXK)-1:0]  rd_addr;
    logic signed [7:0]             rd_data;

    logic [7:0] Cstride, cbase;
    im2col_gen #(.MAXFM(MAXFM), .MAXM(MAXM), .MAXK(MAXK)) dut (
        .clk(clk), .rst(rst), .H(H), .W(W), .C(C), .R(R), .S(S),
        .stride(stride), .P(P), .Q(Q), .Cstride(Cstride), .cbase(cbase),
        .pad_top(pad_top), .pad_left(pad_left),
        .wr_en(wr_en), .wr_addr(wr_addr), .wr_data(wr_data),
        .start(start), .busy(busy), .done(done), .rd_addr(rd_addr), .rd_data(rd_data)
    );

    always #5 clk = ~clk;

    logic signed [7:0] fm [0:MAXFM-1];   // reference feature map (flat, (row*W+col)*C+c)
    int errors = 0, tests = 0;

    task automatic run_one(input int iH, iW, iC, iR, iS, istr, iP, iQ, ipt, ipl);
        tests++;
        H=iH; W=iW; C=iC; R=iR; S=iS; stride=istr; P=iP; Q=iQ; pad_top=ipt; pad_left=ipl;
        Cstride=iC; cbase=0;                       // std/pw: source stride == emit count

        // random feature map + load it
        @(negedge clk);
        for (int row = 0; row < iH; row++)
            for (int col = 0; col < iW; col++)
                for (int ch = 0; ch < iC; ch++) begin
                    logic signed [7:0] t; t = $random;
                    fm[(row*iW + col)*iC + ch] = t;
                    wr_en = 1; wr_addr = (row*iW + col)*iC + ch; wr_data = t; @(negedge clk);
                end
        wr_en = 0;

        // run
        start = 1; @(negedge clk); start = 0;
        wait (done); @(negedge clk);

        // check every cols[m][k] against a direct im2col
        for (int p = 0; p < iP; p++)
          for (int qq = 0; qq < iQ; qq++)
            for (int rr = 0; rr < iR; rr++)
              for (int ss = 0; ss < iS; ss++)
                for (int ch = 0; ch < iC; ch++) begin
                    int m, k, srr, scc; logic signed [7:0] exp;
                    m  = p*iQ + qq;
                    k  = (rr*iS + ss)*iC + ch;
                    srr = p*istr + rr - ipt;
                    scc = qq*istr + ss - ipl;
                    exp = (srr>=0 && srr<iH && scc>=0 && scc<iW) ? fm[(srr*iW+scc)*iC+ch] : 8'sd0;
                    rd_addr = m*MAXK + k; #1;
                    if (rd_data !== exp) begin
                        errors++;
                        if (errors <= 12)
                            $display("MISMATCH cols[m=%0d][k=%0d] (p%0d q%0d r%0d s%0d c%0d): got %0d exp %0d",
                                     m, k, p, qq, rr, ss, ch, rd_data, exp);
                    end
                end
        $display("  im2col H%0d W%0d C%0d  %0dx%0d stride%0d pad(%0d,%0d) -> %0dx%0d %s",
                 iH,iW,iC,iR,iS,istr,ipt,ipl,iP,iQ, (errors==0)?"":"<-- FAIL");
    endtask

    // depthwise: emit ONE channel (C=1) pulled from a Cfull-channel interleaved map.
    task automatic run_dw(input int iH,iW,iCfull,iR,iS,istr,iP,iQ,ipt,ipl,ici);
        tests++;
        H=iH;W=iW;C=8'd1;R=iR;S=iS;stride=istr;P=iP;Q=iQ;pad_top=ipt;pad_left=ipl;
        Cstride=iCfull; cbase=ici;
        @(negedge clk);
        for (int row=0;row<iH;row++) for (int col=0;col<iW;col++) for (int ch=0;ch<iCfull;ch++) begin
            logic signed [7:0] t; t=$random; fm[(row*iW+col)*iCfull+ch]=t;
            wr_en=1; wr_addr=(row*iW+col)*iCfull+ch; wr_data=t; @(negedge clk);
        end
        wr_en=0;
        start=1; @(negedge clk); start=0; wait(done); @(negedge clk);
        for (int p=0;p<iP;p++) for (int qq=0;qq<iQ;qq++) for (int rr=0;rr<iR;rr++) for (int ss=0;ss<iS;ss++) begin
            int m,k,srr,scc; logic signed [7:0] exp;
            m=p*iQ+qq; k=rr*iS+ss; srr=p*istr+rr-ipt; scc=qq*istr+ss-ipl;
            exp = (srr>=0&&srr<iH&&scc>=0&&scc<iW) ? fm[(srr*iW+scc)*iCfull+ici] : 8'sd0;
            rd_addr=m*MAXK+k; #1;
            if (rd_data!==exp) begin errors++; if(errors<=12) $display("DW MISMATCH m%0d k%0d got %0d exp %0d",m,k,rd_data,exp); end
        end
        $display("  im2col-dw H%0d W%0d Cfull%0d ch%0d %0dx%0d s%0d -> %0dx%0d %s",
                 iH,iW,iCfull,ici,iR,iS,istr,iP,iQ,(errors==0)?"":"<-- FAIL");
    endtask

    initial begin
        wr_en=0; start=0; rd_addr=0; Cstride=1; cbase=0;
        @(negedge clk); rst = 0; @(negedge clk);

        run_one(49,10, 1, 10,4, 2, 25,5, 4,1);  // DS-CNN conv1 (SAME, stride 2)
        run_one(25, 5, 1,  3,3, 1, 25,5, 1,1);  // depthwise 3x3 (per channel), SAME
        run_one(25, 5, 64, 1,1, 1, 25,5, 0,0);  // pointwise 1x1
        run_one( 7, 5, 3,  3,3, 2,  4,3, 1,1);  // odd strided SAME case
        run_dw (25, 5, 64, 3,3, 1, 25,5, 1,1, 7);  // extract channel 7 from a 64-ch map

        if (errors == 0)
            $display("PASS: %0d im2col tests match reference (conv1 / depthwise / pointwise / strided)", tests);
        else
            $display("FAIL: %0d mismatched cols elements across %0d tests", errors, tests);
        $finish;
    end
endmodule
