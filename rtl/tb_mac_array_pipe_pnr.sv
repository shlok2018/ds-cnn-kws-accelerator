// tb_mac_array_pipe_pnr.sv
// Self-checking testbench for the deeper-pipelined P&R top. Runs a random 8x8x8
// matmul through the pipelined array, then reads every accumulator back out
// through the 2-stage read-out mux (rd_sel -> wait 2 cycles -> rd_data) and
// checks each against the golden result.
`timescale 1ns/1ps
module tb_mac_array_pipe_pnr;
    localparam int N = 8, K = 8, ACCW = 32, TRIALS = 20;

    logic clk = 0, rst = 1, clr = 0, en = 0;
    logic [N*8-1:0]         a_col, w_row;
    logic [$clog2(N*N)-1:0] rd_sel = 0;
    logic [ACCW-1:0]        rd_data;

    mac_array_8x8_pipe_pnr #(.N(N), .ACCW(ACCW)) dut (
        .clk(clk), .rst(rst), .clr(clr), .en(en),
        .a_col(a_col), .w_row(w_row), .rd_sel(rd_sel), .rd_data(rd_data)
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

        // Load operands (one k per enabled cycle), then flush the MAC pipe.
        @(negedge clk); clr = 1; @(negedge clk); clr = 0;
        for (int k = 0; k < K; k++) begin
            for (int i = 0; i < N; i++) a_col[i*8 +: 8] = A[i][k];
            for (int j = 0; j < N; j++) w_row[j*8 +: 8] = W[k][j];
            en = 1; @(negedge clk);
        end
        en = 0; @(negedge clk);   // flush last product (MAC pipe latency)

        // Read every accumulator out through the 2-stage mux.
        for (int idx = 0; idx < N*N; idx++) begin
            rd_sel = idx;
            @(negedge clk);       // stage-1 capture
            @(negedge clk);       // stage-2 capture -> rd_data valid
            begin
                automatic int got = $signed(rd_data);
                automatic int i = idx / N, j = idx % N;
                if (got !== expected[i][j]) begin
                    errors++;
                    $display("MISMATCH O[%0d][%0d] (rd_sel=%0d): got %0d exp %0d",
                             i, j, idx, got, expected[i][j]);
                end
            end
        end
    endtask

    initial begin
        @(negedge clk); rst = 0;
        for (int t = 0; t < TRIALS; t++) run_trial();
        if (errors == 0)
            $display("PASS: %0d pipelined-PNR 8x8x%0d readback trials match golden", TRIALS, K);
        else
            $display("FAIL: %0d mismatched elements", errors);
        $finish;
    end
endmodule
