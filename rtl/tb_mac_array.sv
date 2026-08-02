// tb_mac_array.sv
// Self-checking testbench for mac_array_8x8: random int8 A (NxK) and W (KxN),
// compute the golden O = A@W in the testbench, stream K cycles into the DUT, and
// compare every output element. This is the functional-correctness gate before
// synthesis -- if the RTL matmul matches the golden across many random trials,
// the compute core is sound.
`timescale 1ns/1ps
module tb_mac_array;
    localparam int N = 8, K = 8, ACCW = 32, TRIALS = 20;

    logic clk = 0, rst = 1, clr = 0, en = 0;
    logic [N*8-1:0]      a_col, w_row;
    logic [N*N*ACCW-1:0] o_flat;

    mac_array_8x8 #(.N(N), .ACCW(ACCW)) dut (
        .clk(clk), .rst(rst), .clr(clr), .en(en),
        .a_col(a_col), .w_row(w_row), .o_flat(o_flat)
    );

    always #5 clk = ~clk;

    // int arrays: reads are unambiguously signed (avoids an iverilog quirk where
    // signed unpacked-array reads inside an automatic task come out unsigned).
    int A [N][K];
    int W [K][N];
    int expected [N][N];
    int errors = 0;

    task automatic run_trial();
        // fill with random int8 values (sign-extended into the int arrays)
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
                    expected[i][j] += A[i][k] * W[k][j];   // int*int, signed
            end

        @(negedge clk); clr = 1; @(negedge clk); clr = 0;   // start a new output
        for (int k = 0; k < K; k++) begin
            for (int i = 0; i < N; i++) a_col[i*8 +: 8] = A[i][k];
            for (int j = 0; j < N; j++) w_row[j*8 +: 8] = W[k][j];
            en = 1; @(negedge clk);
        end
        en = 0; @(negedge clk);

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
            $display("PASS: %0d random 8x8x%0d matmul trials match golden", TRIALS, K);
        else
            $display("FAIL: %0d mismatched elements", errors);
        $finish;
    end
endmodule
