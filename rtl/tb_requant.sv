// tb_requant.sv
// Self-checking testbench for the integer requant/activation lane (Phase 2).
// Directed cases pin the arithmetic (rounding, ReLU, saturation) to hand-computed
// values; random + saturation sweeps then check the RTL against a golden model of
// the same fixed-point formula.
`timescale 1ns/1ps
module tb_requant;
    localparam int ACCW = 32, OW = 8;

    logic signed [ACCW-1:0] acc, bias, mult;
    logic [5:0]             shift;
    logic                   relu;
    logic signed [OW-1:0]   y;

    requant_unit #(.ACCW(ACCW), .OW(OW)) dut (
        .acc(acc), .bias(bias), .mult(mult), .shift(shift), .relu(relu), .y(y));

    int errors = 0, tests = 0;

    function automatic logic signed [OW-1:0] golden(
        input logic signed [ACCW-1:0] a, b, m, input logic [5:0] sh, input logic rl);
        longint x, prod, half, r;
        x    = longint'(a) + longint'(b);
        prod = x * longint'(m);
        half = (sh != 0) ? (longint'(1) <<< (sh - 1)) : 0;
        r    = (prod + half) >>> sh;
        if (rl && r < 0) r = 0;
        if      (r >  127) return  8'sd127;
        else if (r < -127) return -8'sd127;
        else               return r[OW-1:0];
    endfunction

    task automatic chk(input logic signed [ACCW-1:0] a, b, m,
                       input logic [5:0] sh, input logic rl);
        tests++;
        acc = a; bias = b; mult = m; shift = sh; relu = rl; #1;
        if (y !== golden(a, b, m, sh, rl)) begin
            errors++;
            if (errors <= 12)
                $display("MISMATCH acc=%0d bias=%0d mult=%0d shift=%0d relu=%0b : got %0d exp %0d",
                         a, b, m, sh, rl, y, golden(a, b, m, sh, rl));
        end
    endtask

    task automatic directed(input logic signed [ACCW-1:0] a, b, m,
                            input logic [5:0] sh, input logic rl,
                            input logic signed [OW-1:0] exp);
        tests++;
        acc = a; bias = b; mult = m; shift = sh; relu = rl; #1;
        if (y !== exp) begin
            errors++;
            $display("DIRECTED fail: acc=%0d mult=%0d shift=%0d relu=%0b : got %0d exp %0d",
                     a, m, sh, rl, y, exp);
        end
    endtask

    initial begin
        // ---- directed: pin rounding / ReLU / saturation ----
        directed(100,     0, (1<<20), 22, 0,  8'sd25);   // 100/4        = 25
        directed(11,      0, (1<<20), 21, 0,  8'sd6);    // 11/2 = 5.5  -> 6 (round up)
        directed(-100,    0, (1<<20), 22, 1,  8'sd0);    // -25, ReLU   -> 0
        directed(-100,    0, (1<<20), 22, 0, -8'sd25);   // -25, no ReLU
        directed(100000,  0, (1<<20), 20, 0,  8'sd127);  // 100000      -> +sat
        directed(-100000, 0, (1<<20), 20, 0, -8'sd127);  // -100000     -> -sat
        directed(5,       7, 1,        0, 0,  8'sd12);   // shift 0: (5+7)*1 = 12

        // ---- random realistic requants ----
        for (int i = 0; i < 5000; i++) begin
            logic signed [ACCW-1:0] a, b, m; logic [5:0] sh; logic rl;
            a  = $random % (1 << 21);
            b  = $random % 256;
            m  = (1 << 28) + ($random & ((1 << 30) - 1));
            sh = 20 + ($random % 15);
            rl = $random & 1;
            chk(a, b, m, sh, rl);
        end
        // ---- saturation stress (full-range acc) ----
        for (int i = 0; i < 500; i++) begin
            logic signed [ACCW-1:0] a; logic [5:0] sh;
            a  = $random;
            sh = 10 + ($random % 20);
            chk(a, 0, (1 << 25), sh, $random & 1);
        end

        if (errors == 0)
            $display("PASS: %0d requant tests match golden (directed + random + saturation)", tests);
        else
            $display("FAIL: %0d mismatches across %0d tests", errors, tests);
        $finish;
    end
endmodule
