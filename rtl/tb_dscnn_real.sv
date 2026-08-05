// tb_dscnn_real.sv
// End-to-end check: runs the REAL 10-layer DS-CNN through the hardware sequencer
// on real MFCC clips and confirms the hardware's predicted keyword is bit-exact
// with the static-quant software golden. Descriptors/weights/params/clips are
// emitted by sim/dscnn_static.py (EMIT=n) into rtl/gen/*.hex and read here.
`timescale 1ns/1ps
module tb_dscnn_real;
    localparam int NLYR=10, NW=22016, NP=588, NCL=6, INSZ=490;
    localparam int BUF=8192, WMEM=32768, PMEM=1024;

    logic clk=0, rst=1;
    logic [3:0] nlayers;
    logic desc_wr_en; logic [4:0] desc_field; logic [3:0] desc_layer; logic signed [31:0] desc_val;
    logic wmem_wr_en; logic [$clog2(WMEM)-1:0] wmem_addr; logic signed [7:0] wmem_data;
    logic pmem_wr_en; logic [1:0] pmem_sel; logic [$clog2(PMEM)-1:0] pmem_addr; logic signed [31:0] pmem_data;
    logic fm_wr_en; logic [$clog2(BUF)-1:0] fm_wr_addr; logic signed [7:0] fm_wr_data;
    logic start, busy, done, out_in_b; logic [$clog2(BUF)-1:0] rd_addr; logic signed [7:0] rd_data; logic [3:0] pred;

    dscnn_seq #(.BUF(BUF),.WMEM(WMEM),.PMEM(PMEM)) dut (
        .clk(clk),.rst(rst),.nlayers(nlayers),
        .desc_wr_en(desc_wr_en),.desc_field(desc_field),.desc_layer(desc_layer),.desc_val(desc_val),
        .wmem_wr_en(wmem_wr_en),.wmem_addr(wmem_addr),.wmem_data(wmem_data),
        .pmem_wr_en(pmem_wr_en),.pmem_sel(pmem_sel),.pmem_addr(pmem_addr),.pmem_data(pmem_data),
        .fm_wr_en(fm_wr_en),.fm_wr_addr(fm_wr_addr),.fm_wr_data(fm_wr_data),
        .start(start),.busy(busy),.done(done),.out_in_b(out_in_b),.rd_addr(rd_addr),.rd_data(rd_data),.pred(pred));

    always #5 clk = ~clk;

    logic [31:0] desc_f[0:NLYR*17-1];
    logic [7:0]  wmem_f[0:NW-1];
    logic [31:0] pmult_f[0:NP-1], pshift_f[0:NP-1], pbias_f[0:NP-1];
    logic [7:0]  clips_f[0:NCL*INSZ-1];
    logic [3:0]  preds_f[0:NCL-1];
    int errors=0;

    task automatic wdesc(input int f, input int l, input logic signed [31:0] v);
        desc_field=f; desc_layer=l; desc_val=v; desc_wr_en=1; @(negedge clk); desc_wr_en=0; endtask

    initial begin
        desc_wr_en=0; wmem_wr_en=0; pmem_wr_en=0; fm_wr_en=0; start=0; rd_addr=0;
        $readmemh("gen/desc.hex", desc_f);
        $readmemh("gen/wmem.hex", wmem_f);
        $readmemh("gen/pmult.hex", pmult_f);
        $readmemh("gen/pshift.hex", pshift_f);
        $readmemh("gen/pbias.hex", pbias_f);
        $readmemh("gen/clips.hex", clips_f);
        $readmemh("gen/preds.hex", preds_f);
        @(negedge clk); rst=0; @(negedge clk);

        // ---- load descriptors, weights, params (once) ----
        for (int l=0;l<NLYR;l++) for (int f=0;f<17;f++) wdesc(f, l, desc_f[l*17+f]);
        for (int i=0;i<NW;i++) begin wmem_wr_en=1; wmem_addr=i; wmem_data=wmem_f[i]; @(negedge clk); end
        wmem_wr_en=0;
        for (int i=0;i<NP;i++) begin
            pmem_wr_en=1; pmem_sel=0; pmem_addr=i; pmem_data=pmult_f[i]; @(negedge clk);
            pmem_sel=1; pmem_data=pshift_f[i]; @(negedge clk);
            pmem_sel=2; pmem_data=pbias_f[i]; @(negedge clk); end
        pmem_wr_en=0;

        // ---- run each clip through the whole 10-layer net ----
        nlayers=NLYR;
        for (int cl=0; cl<NCL; cl++) begin
            for (int a=0;a<INSZ;a++) begin fm_wr_en=1; fm_wr_addr=a; fm_wr_data=clips_f[cl*INSZ+a]; @(negedge clk); end
            fm_wr_en=0;
            start=1; @(negedge clk); start=0;
            wait(done); @(negedge clk);
            if (pred !== preds_f[cl]) begin
                errors++; $display("CLIP %0d MISMATCH: hw pred=%0d golden=%0d", cl, pred, preds_f[cl]);
            end else
                $display("  clip %0d -> keyword class %0d  (matches golden)", cl, pred);
        end

        if (errors==0) $display("PASS: full 10-layer DS-CNN runs in HW, all %0d predictions bit-exact vs software golden", NCL);
        else           $display("FAIL: %0d clip mispredictions", errors);
        $finish;
    end
endmodule
