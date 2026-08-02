// tb_mac_array_pipe.sv
// Self-checking testbench for the PIPELINED array. Identical to tb_mac_array
// except it drives `en` low for one extra cycle after the K operands, to flush
// the last product through the pipeline before reading the result.
`timescale 1ns/1ps
module tb_mac_array_pipe;
    localparam int N = 8, K = 8, ACCW = 32, TRIALS = 20;

    logic clk = 0, rst = 1, clr = 0, en = 0;
    logic [N*8-1:0]      a_col, w_row;
    logic [N*N*ACCW-1:0] o_flat;

    mac_array_8x8_pipe #(.N(N), .ACCW(ACCW)) dut (
        .clk(clk), .rst(rst), .clr(clr), .en(en),
        .a_col(a_col), .w_row(w_row), .o_flat(o_flat)
    );

    always #5 clk = ~clk;

    int A [N][K];
    int W [K][N];
    int expected [N][N];
    int errors = 0;

    task automatic run_trial();
        for (int i = 0; i < N; i++)
            for (int k = 0; k < K; k++) begin
                logic signed [7:0] t; t = $random; A[i][k] = t;
            end
        for (int k = 0; k < K; k++)
            for (int j = 0; j < N; j++) begin
                logic signed [7:0] t; t = $random; W[k][j] = t;
            end
        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++) begin
                expected[i][j] = 0;
                for (int k = 0; k < K; k++)
                    expected[i][j] += A[i][k] * W[k][j];
            end

        @(negedge clk); clr = 1; @(negedge clk); clr = 0;
        for (int k = 0; k < K; k++) begin
            for (int i = 0; i < N; i++) a_col[i*8 +: 8] = A[i][k];
            for (int j = 0; j < N; j++) w_row[j*8 +: 8] = W[k][j];
            en = 1; @(negedge clk);
        end
        en = 0; @(negedge clk);   // flush the last product (1-cycle pipe latency)

        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++) begin
                automatic int got = $signed(o_flat[(i*N + j)*ACCW +: ACCW]);
                if (got !== expected[i][j]) begin
                    errors++;
                    $display("MISMATCH O[%0d][%0d]: got %0d exp %0d",
                             i, j, got, expected[i][j]);
                end
            end
    endtask

    initial begin
        @(negedge clk); rst = 0;
        for (int t = 0; t < TRIALS; t++) run_trial();
        if (errors == 0)
            $display("PASS: %0d pipelined 8x8x%0d matmul trials match golden", TRIALS, K);
        else
            $display("FAIL: %0d mismatched elements", errors);
        $finish;
    end
endmodule
