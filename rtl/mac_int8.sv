// mac_int8.sv
// One int8 x int8 multiply-accumulate PE, output-stationary.
// The multiplier is 8-bit signed (matches the Timeloop `intmac` component's
// multiplier_width=8 that Accelergy costed). The accumulator is parameterizable
// and made wide enough (default 32b) not to overflow across the streamed K; the
// Timeloop model used a 16b adder, which is a modeling simplification -- the
// multiplier dominates area/energy, and that is what we compare.
module mac_int8 #(parameter int ACCW = 32) (
    input  logic                   clk,
    input  logic                   rst,   // sync reset of accumulator
    input  logic                   clr,   // clear accumulator (start a new output)
    input  logic                   en,    // accumulate a*w this cycle
    input  logic signed [7:0]      a,     // activation
    input  logic signed [7:0]      w,     // weight
    output logic signed [ACCW-1:0] acc
);
    logic signed [15:0] prod;
    assign prod = a * w;                   // signed 8x8 -> 16b product

    always_ff @(posedge clk) begin
        if (rst || clr) acc <= '0;
        else if (en)    acc <= acc + prod; // prod sign-extended into ACCW
    end
endmodule
