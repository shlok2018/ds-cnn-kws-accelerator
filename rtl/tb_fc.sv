// tb_fc.sv
// Self-checking testbench for the classifier tail (avgpool -> requant -> FC ->
// argmax). Checks both the pooled+requant vector vq and the predicted class
// against a golden.
`timescale 1ns/1ps
module tb_fc;
    localparam int MAXIN=8192, MAXC=64, MAXNC=16, MAXM=128, MAXK=64, MAXP=64;

    logic clk=0, rst=1;
    logic [7:0] M,C,NC;
    logic fm_wr_en; logic [$clog2(MAXIN)-1:0] fm_wr_addr; logic signed [7:0] fm_wr_data;
    logic w_wr_en;  logic [$clog2(MAXK*MAXP)-1:0] w_wr_addr; logic signed [7:0] w_wr_data;
    logic signed [31:0] pool_mult; logic [5:0] pool_shift;
    logic fcp_wr_en, fcp_sel; logic [$clog2(MAXNC)-1:0] fcp_addr; logic signed [31:0] fcp_data;
    logic start, busy, done;
    logic [$clog2(MAXC)-1:0] rd_addr; logic signed [7:0] vq_data; logic [3:0] pred;

    fc_engine #(.MAXIN(MAXIN),.MAXC(MAXC),.MAXNC(MAXNC),.MAXM(MAXM),.MAXK(MAXK),.MAXP(MAXP)) dut (
        .clk(clk),.rst(rst),.M(M),.C(C),.NC(NC),
        .fm_wr_en(fm_wr_en),.fm_wr_addr(fm_wr_addr),.fm_wr_data(fm_wr_data),
        .w_wr_en(w_wr_en),.w_wr_addr(w_wr_addr),.w_wr_data(w_wr_data),
        .pool_mult(pool_mult),.pool_shift(pool_shift),
        .fcp_wr_en(fcp_wr_en),.fcp_sel(fcp_sel),.fcp_addr(fcp_addr),.fcp_data(fcp_data),
        .start(start),.busy(busy),.done(done),.rd_addr(rd_addr),.vq_data(vq_data),.pred(pred));

    always #5 clk = ~clk;

    logic signed [7:0] fmv [0:MAXM-1][0:MAXC-1];
    logic signed [7:0] fcw [0:MAXC-1][0:MAXNC-1];
    logic signed [31:0] fmul[0:MAXNC-1], fbia[0:MAXNC-1];
    logic signed [7:0] vqref [0:MAXC-1];
    int errors=0, tests=0;

    function automatic logic signed [7:0] rqpool(input longint acc, input logic signed [31:0] m, input logic [5:0] sh);
        longint half,r; half=(sh!=0)?(longint'(1)<<<(sh-1)):0; r=(acc*m+half)>>>sh;
        if (r<0) r=0; if (r>127) return 8'sd127; else return r[7:0];
    endfunction

    task automatic run_fc(input int iM,iC,iNC);
        tests++;
        M=iM; C=iC; NC=iNC; pool_mult=(1<<20); pool_shift=6'd27;
        for (int m=0;m<iM;m++) for (int c=0;c<iC;c++) begin logic signed[7:0] t; t=$random & 8'h7f; fmv[m][c]=t; end
        for (int c=0;c<iC;c++) for (int o=0;o<iNC;o++) begin logic signed[7:0] t; t=$random; fcw[c][o]=t; end
        for (int o=0;o<iNC;o++) begin fmul[o]=(1<<12)+(o*7); fbia[o]=($random%100000); end

        // golden: pool -> requant -> fc -> argmax
        for (int c=0;c<iC;c++) begin
            longint s; s=0; for (int m=0;m<iM;m++) s+=fmv[m][c];
            vqref[c]=rqpool(s, pool_mult, pool_shift);
        end
        begin
            longint best; int bp;
            best = 64'sh8000000000000000; bp=0;
            for (int o=0;o<iNC;o++) begin
                longint raw,lg; raw=0;
                for (int c=0;c<iC;c++) raw += vqref[c]*fcw[c][o];
                lg = raw*fmul[o] + fbia[o];
                if (lg > best) begin best=lg; bp=o; end
            end

            // load
            @(negedge clk);
            for (int m=0;m<iM;m++) for (int c=0;c<iC;c++) begin fm_wr_en=1; fm_wr_addr=m*iC+c; fm_wr_data=fmv[m][c]; @(negedge clk); end
            fm_wr_en=0;
            for (int c=0;c<iC;c++) for (int o=0;o<iNC;o++) begin w_wr_en=1; w_wr_addr=c*MAXP+o; w_wr_data=fcw[c][o]; @(negedge clk); end
            w_wr_en=0;
            for (int o=0;o<iNC;o++) begin fcp_wr_en=1; fcp_sel=0; fcp_addr=o; fcp_data=fmul[o]; @(negedge clk); fcp_sel=1; fcp_data=fbia[o]; @(negedge clk); end
            fcp_wr_en=0;

            start=1; @(negedge clk); start=0; wait(done); @(negedge clk);

            // check vq vector
            for (int c=0;c<iC;c++) begin
                rd_addr=c; #1;
                if (vq_data!==vqref[c]) begin errors++; if(errors<=8) $display("VQ MISMATCH c=%0d got %0d exp %0d",c,vq_data,vqref[c]); end
            end
            if (pred!==bp[3:0]) begin errors++; $display("PRED MISMATCH got %0d exp %0d",pred,bp); end
            $display("  fc M%0d C%0d NC%0d : vq+argmax  pred(dut)=%0d pred(gold)=%0d %s",
                     iM,iC,iNC,pred,bp,(errors==0)?"":"<-- FAIL");
        end
    endtask

    initial begin
        fm_wr_en=0; w_wr_en=0; fcp_wr_en=0; start=0; rd_addr=0;
        @(negedge clk); rst=0; @(negedge clk);
        run_fc(10, 8,  4);      // small
        run_fc(125,64, 12);     // DS-CNN FC: 125 positions, 64 ch, 12 classes
        if (errors==0) $display("PASS: %0d fc-engine tests match golden (avgpool->requant->FC->argmax in HW)", tests);
        else           $display("FAIL: %0d mismatches across %0d tests", errors, tests);
        $finish;
    end
endmodule
