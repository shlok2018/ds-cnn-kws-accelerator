// tb_seq.sv
// Self-checking testbench for the multi-layer sequencer. Runs a 3-layer chain
// (3x3 standard conv -> pointwise -> pointwise) entirely in hardware and checks
// the final output map against a golden that runs the same three layers in
// sequence -- verifying the descriptor walk and the ping-pong buffering.
`timescale 1ns/1ps
module tb_seq;
    localparam int BUF=2048, WMEM=2048, PMEM=128, MAXM=128, MAXK=64, MAXP=64;

    logic clk=0, rst=1;
    logic [3:0] nlayers;
    logic desc_wr_en; logic [3:0] desc_field, desc_layer; logic signed [31:0] desc_val;
    logic wmem_wr_en; logic [$clog2(WMEM)-1:0] wmem_addr; logic signed [7:0] wmem_data;
    logic pmem_wr_en; logic [1:0] pmem_sel; logic [$clog2(PMEM)-1:0] pmem_addr; logic signed [31:0] pmem_data;
    logic fm_wr_en; logic [$clog2(BUF)-1:0] fm_wr_addr; logic signed [7:0] fm_wr_data;
    logic start, busy, done, out_in_b;
    logic [$clog2(BUF)-1:0] rd_addr; logic signed [7:0] rd_data;

    dscnn_seq #(.BUF(BUF),.WMEM(WMEM),.PMEM(PMEM),.MAXM(MAXM),.MAXK(MAXK),.MAXP(MAXP)) dut (
        .clk(clk),.rst(rst),.nlayers(nlayers),
        .desc_wr_en(desc_wr_en),.desc_field(desc_field),.desc_layer(desc_layer),.desc_val(desc_val),
        .wmem_wr_en(wmem_wr_en),.wmem_addr(wmem_addr),.wmem_data(wmem_data),
        .pmem_wr_en(pmem_wr_en),.pmem_sel(pmem_sel),.pmem_addr(pmem_addr),.pmem_data(pmem_data),
        .fm_wr_en(fm_wr_en),.fm_wr_addr(fm_wr_addr),.fm_wr_data(fm_wr_data),
        .start(start),.busy(busy),.done(done),.out_in_b(out_in_b),.rd_addr(rd_addr),.rd_data(rd_data));

    always #5 clk = ~clk;

    // reference storage
    logic signed [7:0]  wref [0:WMEM-1];
    logic signed [31:0] pmul_r[0:PMEM-1], pbia_r[0:PMEM-1]; logic [5:0] psh_r[0:PMEM-1];
    logic signed [7:0]  gA [0:BUF-1], gB [0:BUF-1];
    // per-layer descriptor (host copy)
    int dH[0:3],dW[0:3],dC[0:3],dR[0:3],dS[0:3],dstr[0:3],dCo[0:3],dP[0:3],dQ[0:3],dpt[0:3],dpl[0:3],drl[0:3],dwo[0:3],dpo[0:3];
    int NL, errors=0;

    function automatic logic signed [7:0] rq(input longint acc,m,b,input logic[5:0] sh,input int rl);
        longint x,half,r; x=acc+b; half=(sh!=0)?(longint'(1)<<<(sh-1)):0; r=(x*m+half)>>>sh;
        if (rl && r<0) r=0; if (r>127) return 8'sd127; else if (r<-127) return -8'sd127; else return r[7:0];
    endfunction

    // one golden layer: src buffer -> dst buffer (A<->B ping-pong by li parity)
    task automatic golden_layer(input int li);
        int H,W,C,R,S,st,Co,P,Q,pt,pl,rl,wo,po; logic srcA;
        H=dH[li];W=dW[li];C=dC[li];R=dR[li];S=dS[li];st=dstr[li];Co=dCo[li];
        P=dP[li];Q=dQ[li];pt=dpt[li];pl=dpl[li];rl=drl[li];wo=dwo[li];po=dpo[li];
        srcA = (li[0]==0);
        for (int p=0;p<P;p++) for (int q=0;q<Q;q++) for (int o=0;o<Co;o++) begin
            longint acc; acc=0;
            for (int r=0;r<R;r++) for (int s=0;s<S;s++) for (int c=0;c<C;c++) begin
                int sr,sc,kk; logic signed [7:0] a;
                sr=p*st+r-pt; sc=q*st+s-pl; kk=(r*S+s)*C+c;
                if (sr>=0&&sr<H&&sc>=0&&sc<W) a = srcA ? gA[(sr*W+sc)*C+c] : gB[(sr*W+sc)*C+c];
                else a = 8'sd0;
                acc += a * wref[wo + kk*Co + o];
            end
            if (srcA) gB[(p*Q+q)*Co+o] = rq(acc, pmul_r[po+o], pbia_r[po+o], psh_r[po+o], rl);
            else      gA[(p*Q+q)*Co+o] = rq(acc, pmul_r[po+o], pbia_r[po+o], psh_r[po+o], rl);
        end
    endtask

    task automatic wdesc(input int f,l,v); desc_field=f; desc_layer=l; desc_val=v; desc_wr_en=1; @(negedge clk); desc_wr_en=0; endtask

    initial begin
        desc_wr_en=0; wmem_wr_en=0; pmem_wr_en=0; fm_wr_en=0; start=0; rd_addr=0;
        @(negedge clk); rst=0; @(negedge clk);

        // ---- define a 3-layer net: std conv 3x3 (4->8), pw (8->8), pw (8->4) ----
        NL=3; nlayers=3;
        //            H W C R S st Co P Q pt pl rl  wo   po
        dH[0]=5;dW[0]=4;dC[0]=4;dR[0]=3;dS[0]=3;dstr[0]=1;dCo[0]=8;dP[0]=5;dQ[0]=4;dpt[0]=1;dpl[0]=1;drl[0]=1;dwo[0]=0;   dpo[0]=0;
        dH[1]=5;dW[1]=4;dC[1]=8;dR[1]=1;dS[1]=1;dstr[1]=1;dCo[1]=8;dP[1]=5;dQ[1]=4;dpt[1]=0;dpl[1]=0;drl[1]=1;dwo[1]=288; dpo[1]=8;
        dH[2]=5;dW[2]=4;dC[2]=8;dR[2]=1;dS[2]=1;dstr[2]=1;dCo[2]=4;dP[2]=5;dQ[2]=4;dpt[2]=0;dpl[2]=0;drl[2]=1;dwo[2]=352; dpo[2]=16;
        // wo: L0 K=3*3*4=36, Co=8 -> 288 weights [0..287]; L1 K=8,Co=8 ->64 [288..351]; L2 K=8,Co=4 ->32 [352..383]
        // po: L0 8 ch [0..7]; L1 8 ch [8..15]; L2 4 ch [16..19]

        // random weights + params + input map
        for (int i=0;i<384;i++) begin logic signed[7:0] t; t=$random; wref[i]=t; end
        for (int i=0;i<20;i++) begin pmul_r[i]=(1<<20)+(i*53); psh_r[i]=6'd33; pbia_r[i]=($random%2048); end
        for (int a=0;a<5*4*4;a++) begin logic signed[7:0] t; t=$random; gA[a]=t; end

        // golden: run the 3 layers
        golden_layer(0); golden_layer(1); golden_layer(2);   // final in gB

        // ---- load descriptors ----
        for (int l=0;l<NL;l++) begin
            wdesc(0,l,dH[l]); wdesc(1,l,dW[l]); wdesc(2,l,dC[l]); wdesc(3,l,dR[l]); wdesc(4,l,dS[l]);
            wdesc(5,l,dstr[l]); wdesc(6,l,dCo[l]); wdesc(7,l,dP[l]); wdesc(8,l,dQ[l]); wdesc(9,l,dpt[l]);
            wdesc(10,l,dpl[l]); wdesc(11,l,drl[l]); wdesc(12,l,dwo[l]); wdesc(13,l,dpo[l]);
        end
        // ---- load weights, params, input map ----
        for (int i=0;i<384;i++) begin wmem_wr_en=1; wmem_addr=i; wmem_data=wref[i]; @(negedge clk); end
        wmem_wr_en=0;
        for (int i=0;i<20;i++) begin
            pmem_wr_en=1; pmem_sel=0; pmem_addr=i; pmem_data=pmul_r[i]; @(negedge clk);
            pmem_sel=1; pmem_data=psh_r[i]; @(negedge clk); pmem_sel=2; pmem_data=pbia_r[i]; @(negedge clk); end
        pmem_wr_en=0;
        for (int a=0;a<5*4*4;a++) begin fm_wr_en=1; fm_wr_addr=a; fm_wr_data=gA[a]; @(negedge clk); end
        fm_wr_en=0;

        // ---- run + check final map (5x4x4) ----
        start=1; @(negedge clk); start=0; wait(done); @(negedge clk);
        for (int a=0;a<5*4*4;a++) begin
            rd_addr=a; #1;
            if (rd_data !== gB[a]) begin errors++; if(errors<=12) $display("MISMATCH out[%0d]: got %0d exp %0d",a,rd_data,gB[a]); end
        end
        $display("final buffer: %s (out_in_b=%0b)", (out_in_b)?"B":"A", out_in_b);
        if (errors==0) $display("PASS: 3-layer DS-CNN-style chain runs in HW, final map matches golden");
        else           $display("FAIL: %0d mismatched output elements", errors);
        $finish;
    end
endmodule
