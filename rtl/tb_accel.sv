// tb_accel.sv -- self-checking testbench for the full accelerator.
// Loads random A and W through the write port, pulses start, waits for done,
// reads O back through the read port, and checks it against a golden matmul.
`timescale 1ns/1ps
module tb_accel;
    localparam int N = 8, K = 8, ACCW = 32, TRIALS = 20;

    logic                   clk = 0, rst = 1;
    logic                   wr_en = 0, wr_is_w = 0;
    logic [$clog2(N*K)-1:0] wr_addr = 0;
    logic signed [7:0]      wr_data = 0;
    logic                   start = 0, busy, done;
    logic [$clog2(N*N)-1:0] rd_addr = 0;
    logic signed [ACCW-1:0] rd_data;

    accel_top #(.N(N), .K(K), .ACCW(ACCW)) dut (.*);

    always #5 clk = ~clk;

    int A [N][K], W [K][N], expected [N][N], errors = 0;

    task automatic run_trial();
        logic signed [7:0] t;
        // random operands + golden result
        for (int i = 0; i < N; i++) for (int k = 0; k < K; k++) begin t = $random; A[i][k] = t; end
        for (int k = 0; k < K; k++) for (int j = 0; j < N; j++) begin t = $random; W[k][j] = t; end
        for (int i = 0; i < N; i++) for (int j = 0; j < N; j++) begin
            expected[i][j] = 0;
            for (int k = 0; k < K; k++) expected[i][j] += A[i][k] * W[k][j];
        end

        // load A then W, one byte per cycle
        @(negedge clk);
        for (int i = 0; i < N; i++) for (int k = 0; k < K; k++) begin
            wr_en <= 1; wr_is_w <= 0; wr_addr <= i*K + k; wr_data <= A[i][k]; @(negedge clk);
        end
        for (int k = 0; k < K; k++) for (int j = 0; j < N; j++) begin
            wr_en <= 1; wr_is_w <= 1; wr_addr <= k*N + j; wr_data <= W[k][j]; @(negedge clk);
        end
        wr_en <= 0;

        // kick off, wait for completion
        start <= 1; @(negedge clk); start <= 0;
        wait (done); @(negedge clk);

        // read back and check
        for (int i = 0; i < N; i++) for (int j = 0; j < N; j++) begin
            rd_addr <= i*N + j; @(negedge clk);
            if ($signed(rd_data) !== expected[i][j]) begin
                errors++;
                $display("MISMATCH O[%0d][%0d]: got %0d exp %0d", i, j, $signed(rd_data), expected[i][j]);
            end
        end
    endtask

    initial begin
        @(negedge clk); rst = 0;
        for (int t = 0; t < TRIALS; t++) run_trial();
        if (errors == 0)
            $display("PASS: %0d accelerator matmul trials match golden (load/start/done/read)", TRIALS);
        else
            $display("FAIL: %0d mismatched elements", errors);
        $finish;
    end
endmodule
