// mac_int8_pipe.sv
// Pipelined int8 MAC PE. A register between the multiplier and the accumulate-
// adder splits the (multiply + add) critical path into two shorter stages, so
// the clock can run faster. Cost: one cycle of latency -- the product of the
// operands presented on cycle t is accumulated on cycle t+1, and `en` is
// pipelined (en_q) so the accumulate is gated by the right cycle's enable.
module mac_int8_pipe #(parameter int ACCW = 32) (
    input  logic                   clk,
    input  logic                   rst,
    input  logic                   clr,   // clear accumulator + flush pipeline
    input  logic                   en,    // present a*w this cycle
    input  logic signed [7:0]      a,
    input  logic signed [7:0]      w,
    output logic signed [ACCW-1:0] acc
);
    logic signed [15:0] prod_q;    // stage 1: registered product
    logic               en_q;      // enable delayed to align with prod_q

    always_ff @(posedge clk) begin
        if (rst || clr) begin prod_q <= '0; en_q <= 1'b0; end
        else            begin prod_q <= a * w; en_q <= en; end
    end

    always_ff @(posedge clk) begin
        if (rst || clr) acc <= '0;
        else if (en_q)  acc <= acc + prod_q;   // stage 2: accumulate
    end
endmodule
