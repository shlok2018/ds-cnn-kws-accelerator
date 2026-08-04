// tb_gemm.sv
// Self-checking testbench for the tiling GEMM engine (Phase 1 of the HW sequencer).
// Runs random (M x K)*(K x P) int8 matmuls -- including the exact DS-CNN layer
// shapes and awkward partial-tile sizes -- through gemm_top and checks every
// output element against a golden matmul computed here.
`timescale 1ns/1ps
module tb_gemm;
    localparam int N = 8, ACCW = 32, MAXM = 128, MAXK = 64, MAXP = 64;

    logic clk = 0, rst = 1;
    logic [$clog2(MAXM+1)-1:0]    m_dim;
    logic [$clog2(MAXK+1)-1:0]    k_dim;
    logic [$clog2(MAXP+1)-1:0]    p_dim;
    logic                         wr_en, wr_is_w;
    logic [$clog2(MAXM*MAXK)-1:0] wr_addr;
    logic signed [7:0]            wr_data;
    logic                         start, busy, done;
    logic [$clog2(MAXM*MAXP)-1:0] rd_addr;
    logic signed [ACCW-1:0]       rd_data;

    gemm_top #(.N(N), .ACCW(ACCW), .MAXM(MAXM), .MAXK(MAXK), .MAXP(MAXP)) dut (
        .clk(clk), .rst(rst), .m_dim(m_dim), .k_dim(k_dim), .p_dim(p_dim),
        .wr_en(wr_en), .wr_is_w(wr_is_w), .wr_addr(wr_addr), .wr_data(wr_data),
        .start(start), .busy(busy), .done(done), .rd_addr(rd_addr), .rd_data(rd_data)
    );

    always #5 clk = ~clk;

    logic signed [7:0] Aarr [0:MAXM-1][0:MAXK-1];
    logic signed [7:0] Warr [0:MAXK-1][0:MAXP-1];
    int                Oref [0:MAXM-1][0:MAXP-1];
    int errors = 0, tests = 0;

    task automatic run_one(input int M, input int K, input int P);
        tests++;
        // random operands + golden matmul
        for (int m = 0; m < M; m++)
            for (int kk = 0; kk < K; kk++) begin
                logic signed [7:0] t; t = $random; Aarr[m][kk] = t;
            end
        for (int kk = 0; kk < K; kk++)
            for (int p = 0; p < P; p++) begin
                logic signed [7:0] t; t = $random; Warr[kk][p] = t;
            end
        for (int m = 0; m < M; m++)
            for (int p = 0; p < P; p++) begin
                int s; s = 0;
                for (int kk = 0; kk < K; kk++) s += Aarr[m][kk] * Warr[kk][p];
                Oref[m][p] = s;
            end

        m_dim = M; k_dim = K; p_dim = P;

        // load A then W (one byte per cycle)
        @(negedge clk);
        for (int m = 0; m < M; m++)
            for (int kk = 0; kk < K; kk++) begin
                wr_en = 1; wr_is_w = 0; wr_addr = m*MAXK + kk; wr_data = Aarr[m][kk];
                @(negedge clk);
            end
        for (int kk = 0; kk < K; kk++)
            for (int p = 0; p < P; p++) begin
                wr_en = 1; wr_is_w = 1; wr_addr = kk*MAXP + p; wr_data = Warr[kk][p];
                @(negedge clk);
            end
        wr_en = 0;

        // run
        start = 1; @(negedge clk); start = 0;
        wait (done); @(negedge clk);

        // check every output element
        for (int m = 0; m < M; m++)
            for (int p = 0; p < P; p++) begin
                rd_addr = m*MAXP + p; #1;
                if ($signed(rd_data) !== Oref[m][p]) begin
                    errors++;
                    if (errors <= 10)
                        $display("MISMATCH (M=%0d K=%0d P=%0d) O[%0d][%0d]: got %0d exp %0d",
                                 M, K, P, m, p, $signed(rd_data), Oref[m][p]);
                end
            end
        $display("  checked %0dx%0d x %0dx%0d GEMM (%0d elems)%s",
                 M, K, K, P, M*P, (errors==0) ? "" : "  <-- FAIL");
    endtask

    initial begin
        wr_en = 0; start = 0; rd_addr = 0;
        @(negedge clk); rst = 0; @(negedge clk);

        run_one(8,   8,  8);    // single tile
        run_one(13,  7,  11);   // partial tiles on all three dims, K<8
        run_one(16,  64, 16);   // exact multiples of 8
        run_one(125, 40, 64);   // DS-CNN conv1 : (P*Q x R*S*Cin)*(.. x Cout)
        run_one(125, 9,  1);    // DS-CNN depthwise: (P*Q x R*S)*(R*S x 1)
        run_one(125, 64, 64);   // DS-CNN pointwise: (P*Q x Cin)*(Cin x Cout)
        run_one(1,   64, 12);   // DS-CNN FC : (1 x 64)*(64 x 12)

        if (errors == 0)
            $display("PASS: %0d tiling-GEMM tests match golden (incl. all DS-CNN layer shapes)", tests);
        else
            $display("FAIL: %0d mismatched elements across %0d tests", errors, tests);
        $finish;
    end
endmodule
