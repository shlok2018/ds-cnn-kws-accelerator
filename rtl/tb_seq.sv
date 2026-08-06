// tb_seq.sv
// Self-checking testbench for the full multi-layer sequencer with per-type
// dispatch. Runs a DS-CNN-shaped chain -- 3x3 standard conv -> depthwise 3x3 ->
// pointwise -> FC(argmax) -- entirely in hardware. Checks (A) the feature map
// after the conv/dw/pw stack and (B) the final predicted class, both vs a golden
// that runs the same layers in sequence.
`timescale 1ns/1ps
module tb_seq;
    localparam int BUF=2048, WMEM=2048, PMEM=128, MAXM=128, MAXK=64, MAXP=64, MAXC=64, MAXRS=16, MAXNC=16;

    logic clk=0, rst=1;
    logic [3:0] nlayers;
    logic desc_wr_en; logic [4:0] desc_field; logic [3:0] desc_layer; logic signed [31:0] desc_val;
    logic wmem_wr_en; logic [$clog2(WMEM)-1:0] wmem_addr; logic signed [7:0] wmem_data;
    logic pmem_wr_en; logic [1:0] pmem_sel; logic [$clog2(PMEM)-1:0] pmem_addr; logic signed [31:0] pmem_data;
    logic fm_wr_en; logic [$clog2(BUF)-1:0] fm_wr_addr; logic signed [7:0] fm_wr_data;
    logic start, busy, done, out_in_b; logic [$clog2(BUF)-1:0] rd_addr; logic signed [7:0] rd_data; logic [3:0] pred;

    dscnn_seq #(.BUF(BUF),.WMEM(WMEM),.PMEM(PMEM),.MAXM(MAXM),.MAXK(MAXK),.MAXP(MAXP),.MAXC(MAXC),.MAXRS(MAXRS),.MAXNC(MAXNC)) dut (
        .clk(clk),.rst(rst),.nlayers(nlayers),
        .desc_wr_en(desc_wr_en),.desc_field(desc_field),.desc_layer(desc_layer),.desc_val(desc_val),
        .wmem_wr_en(wmem_wr_en),.wmem_addr(wmem_addr),.wmem_data(wmem_data),
        .pmem_wr_en(pmem_wr_en),.pmem_sel(pmem_sel),.pmem_addr(pmem_addr),.pmem_data(pmem_data),
        .fm_wr_en(fm_wr_en),.fm_wr_addr(fm_wr_addr),.fm_wr_data(fm_wr_data),
        .start(start),.busy(busy),.done(done),.out_in_b(out_in_b),.rd_addr(rd_addr),.rd_data(rd_data),.pred(pred));

    always #5 clk = ~clk;

    logic signed [7:0]  wref [0:WMEM-1];
    logic signed [31:0] pmul_r[0:PMEM-1], pbia_r[0:PMEM-1]; logic [5:0] psh_r[0:PMEM-1];
    logic signed [7:0]  gA [0:BUF-1], gB [0:BUF-1];
    int errors=0;
    logic signed [31:0] POOLM = (1<<20); logic [5:0] POOLS = 6'd27;
    // layer geometry: L0 conv, L1 dw, L2 pw, L3 fc
    int dH[0:3],dW[0:3],dC[0:3],dR[0:3],dS[0:3],dst_[0:3],dCo[0:3],dP[0:3],dQ[0:3],dpt[0:3],dpl[0:3],drl[0:3],dtyp[0:3],dwo[0:3],dpo[0:3];

    function automatic logic signed [7:0] rq(input longint acc,m,b,input logic[5:0] sh,input int rl);
        longint x,half,r; x=acc+b; half=(sh!=0)?(longint'(1)<<<(sh-1)):0; r=(x*m+half)>>>sh;
        if (rl&&r<0) r=0; if (r>127) return 8'sd127; else if (r<-127) return -8'sd127; else return r[7:0];
    endfunction

    task automatic gld_conv(input int li);   // std/pw conv, src->dst by parity
        int H,W,C,R,S,st,Co,P,Q,pt,pl,rl,wo,po; logic srcA;
        H=dH[li];W=dW[li];C=dC[li];R=dR[li];S=dS[li];st=dst_[li];Co=dCo[li];P=dP[li];Q=dQ[li];
        pt=dpt[li];pl=dpl[li];rl=drl[li];wo=dwo[li];po=dpo[li]; srcA=(li[0]==0);
        for (int p=0;p<P;p++) for (int q=0;q<Q;q++) for (int o=0;o<Co;o++) begin
            longint acc; acc=0;
            for (int r=0;r<R;r++) for (int s=0;s<S;s++) for (int c=0;c<C;c++) begin
                int sr,sc; logic signed[7:0] a; sr=p*st+r-pt; sc=q*st+s-pl;
                a=(sr>=0&&sr<H&&sc>=0&&sc<W)?(srcA?gA[(sr*W+sc)*C+c]:gB[(sr*W+sc)*C+c]):8'sd0;
                acc += a*wref[wo+((r*S+s)*C+c)*Co+o];
            end
            if (srcA) gB[(p*Q+q)*Co+o]=rq(acc,pmul_r[po+o],pbia_r[po+o],psh_r[po+o],rl);
            else      gA[(p*Q+q)*Co+o]=rq(acc,pmul_r[po+o],pbia_r[po+o],psh_r[po+o],rl);
        end
    endtask

    task automatic gld_dw(input int li);     // depthwise, src->dst by parity
        int H,W,C,R,S,st,P,Q,pt,pl,wo,po; logic srcA;
        H=dH[li];W=dW[li];C=dC[li];R=dR[li];S=dS[li];st=dst_[li];P=dP[li];Q=dQ[li];
        pt=dpt[li];pl=dpl[li];wo=dwo[li];po=dpo[li]; srcA=(li[0]==0);
        for (int c=0;c<C;c++) for (int p=0;p<P;p++) for (int q=0;q<Q;q++) begin
            longint acc; acc=0;
            for (int r=0;r<R;r++) for (int s=0;s<S;s++) begin
                int sr,sc; logic signed[7:0] a; sr=p*st+r-pt; sc=q*st+s-pl;
                a=(sr>=0&&sr<H&&sc>=0&&sc<W)?(srcA?gA[(sr*W+sc)*C+c]:gB[(sr*W+sc)*C+c]):8'sd0;
                acc += a*wref[wo+c*R*S+(r*S+s)];
            end
            if (srcA) gB[(p*Q+q)*C+c]=rq(acc,pmul_r[po+c],pbia_r[po+c],psh_r[po+c],1);
            else      gA[(p*Q+q)*C+c]=rq(acc,pmul_r[po+c],pbia_r[po+c],psh_r[po+c],1);
        end
    endtask

    function automatic int gld_fc(input int li);   // reads src (parity), returns argmax
        int H,W,C,NC,wo,po,M; logic srcA; longint best; int bp;
        logic signed [7:0] vq [0:MAXC-1];
        H=dH[li];W=dW[li];C=dC[li];NC=dCo[li];wo=dwo[li];po=dpo[li]; M=H*W; srcA=(li[0]==0);
        for (int c=0;c<C;c++) begin longint s; s=0;
            for (int m=0;m<M;m++) s += srcA?gA[m*C+c]:gB[m*C+c];
            vq[c]=rq(s,POOLM,0,POOLS,1); end
        best=64'sh8000000000000000; bp=0;
        for (int o=0;o<NC;o++) begin longint raw,lg; raw=0;
            for (int c=0;c<C;c++) raw += vq[c]*wref[wo+c*NC+o];
            lg = raw*pmul_r[po+o] + pbia_r[po+o];
            if (lg>best) begin best=lg; bp=o; end end
        return bp;
    endfunction

    task automatic wdesc(input int f,l,v); desc_field=f; desc_layer=l; desc_val=v; desc_wr_en=1; @(negedge clk); desc_wr_en=0; endtask
    task automatic load_input(); for (int a=0;a<dH[0]*dW[0]*dC[0];a++) begin fm_wr_en=1; fm_wr_addr=a; fm_wr_data=gA[a]; @(negedge clk); end fm_wr_en=0; endtask

    initial begin
        desc_wr_en=0; wmem_wr_en=0; pmem_wr_en=0; fm_wr_en=0; start=0; rd_addr=0;
        @(negedge clk); rst=0; @(negedge clk);

        //          H W C R S st Co P Q pt pl rl typ wo   po
        dH[0]=5;dW[0]=4;dC[0]=4;dR[0]=3;dS[0]=3;dst_[0]=1;dCo[0]=8;dP[0]=5;dQ[0]=4;dpt[0]=1;dpl[0]=1;drl[0]=1;dtyp[0]=0;dwo[0]=0;   dpo[0]=0;
        dH[1]=5;dW[1]=4;dC[1]=8;dR[1]=3;dS[1]=3;dst_[1]=1;dCo[1]=8;dP[1]=5;dQ[1]=4;dpt[1]=1;dpl[1]=1;drl[1]=1;dtyp[1]=1;dwo[1]=288; dpo[1]=8;
        dH[2]=5;dW[2]=4;dC[2]=8;dR[2]=1;dS[2]=1;dst_[2]=1;dCo[2]=8;dP[2]=5;dQ[2]=4;dpt[2]=0;dpl[2]=0;drl[2]=1;dtyp[2]=0;dwo[2]=360; dpo[2]=16;
        dH[3]=5;dW[3]=4;dC[3]=8;dR[3]=1;dS[3]=1;dst_[3]=1;dCo[3]=4;dP[3]=0;dQ[3]=0;dpt[3]=0;dpl[3]=0;drl[3]=0;dtyp[3]=2;dwo[3]=424; dpo[3]=24;
        // weights: L0 36*8=288[0..287]; L1 8*9=72[288..359]; L2 8*8=64[360..423]; L3 8*4=32[424..455]
        // params : L0 ch0..7; L1 ch8..15; L2 ch16..23; L3 class24..27

        for (int i=0;i<456;i++) begin logic signed[7:0] t; t=$random; wref[i]=t; end
        for (int i=0;i<24;i++)  begin pmul_r[i]=(1<<20)+(i*29); psh_r[i]=6'd33; pbia_r[i]=($random%2048); end   // conv/dw/pw
        for (int i=24;i<28;i++) begin pmul_r[i]=(1<<12)+(i*5);  psh_r[i]=6'd0;  pbia_r[i]=($random%100000); end  // fc classes
        for (int a=0;a<5*4*4;a++) begin logic signed[7:0] t; t=$random; gA[a]=t; end

        // golden: conv -> dw -> pw (map in gB) ; then fc -> pred
        gld_conv(0); gld_dw(1); gld_conv(2);
        begin int gp; gp = gld_fc(3);

        // ---- load descriptors (all 4) ----
        for (int l=0;l<4;l++) begin
            wdesc(0,l,dH[l]);wdesc(1,l,dW[l]);wdesc(2,l,dC[l]);wdesc(3,l,dR[l]);wdesc(4,l,dS[l]);
            wdesc(5,l,dst_[l]);wdesc(6,l,dCo[l]);wdesc(7,l,dP[l]);wdesc(8,l,dQ[l]);wdesc(9,l,dpt[l]);
            wdesc(10,l,dpl[l]);wdesc(11,l,drl[l]);wdesc(12,l,dwo[l]);wdesc(13,l,dpo[l]);wdesc(14,l,dtyp[l]);
            wdesc(15,l,POOLM);wdesc(16,l,POOLS);
        end
        for (int i=0;i<456;i++) begin wmem_wr_en=1; wmem_addr=i; wmem_data=wref[i]; @(negedge clk); end
        wmem_wr_en=0;
        for (int i=0;i<28;i++) begin
            pmem_wr_en=1; pmem_sel=0; pmem_addr=i; pmem_data=pmul_r[i]; @(negedge clk);
            pmem_sel=1; pmem_data=psh_r[i]; @(negedge clk); pmem_sel=2; pmem_data=pbia_r[i]; @(negedge clk); end
        pmem_wr_en=0;

        // ---- Test A: conv->dw->pw (nlayers=3), check output map (5x4x8) ----
        load_input(); nlayers=3;
        start=1; @(negedge clk); start=0; wait(done); @(negedge clk);
        for (int a=0;a<5*4*8;a++) begin rd_addr=a; @(posedge clk); #1;  // registered (BRAM) read: 1-cycle latency
            if (rd_data!==gB[a]) begin errors++; if(errors<=8) $display("MAP MISMATCH [%0d] got %0d exp %0d",a,rd_data,gB[a]); end end
        $display("  Test A conv->dw->pw map: %s", (errors==0)?"match":"FAIL");

        // ---- Test B: add FC (nlayers=4), check predicted class ----
        load_input(); nlayers=4;
        start=1; @(negedge clk); start=0; wait(done); @(negedge clk);
        if (pred!==gp[3:0]) begin errors++; $display("PRED MISMATCH got %0d exp %0d",pred,gp); end
        $display("  Test B full conv->dw->pw->fc: pred(dut)=%0d pred(gold)=%0d %s", pred,gp,(pred==gp)?"":"<-- FAIL");

        if (errors==0) $display("PASS: full DS-CNN-style chain (conv/dw/pw/fc) runs in HW, map+prediction match golden");
        else           $display("FAIL: %0d errors", errors);
        end
        $finish;
    end
endmodule
