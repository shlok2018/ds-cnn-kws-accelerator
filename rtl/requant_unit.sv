// requant_unit.sv
// Phase 2 of the hardware sequencer: the integer requantization + activation lane
// that sits between layers. It replaces the Python float dequant->ReLU->requant
// chain (run_dscnn_on_accel.conv_layer) with the standard fixed-point integer
// requant real accelerators use, so it is bit-exact by construction:
//
//   x = acc + bias                         (int32 accumulator + per-channel bias)
//   r = round( x * mult / 2^shift )        (per-channel fixed-point scale)
//   y = clip( relu ? max(r,0) : r , -127, 127 )   -> int8 for the next layer
//
// `mult`/`shift`/`bias` are per-output-channel constants the layer sequencer will
// supply (Phase 4). Rounding is round-half-up via a (1<<(shift-1)) bias before an
// arithmetic right shift. Combinational; register downstream if timing needs it.
module requant_unit #(parameter int ACCW = 32, parameter int OW = 8) (
    input  logic signed [ACCW-1:0]   acc,     // GEMM accumulator (int32)
    input  logic signed [ACCW-1:0]   bias,    // folded bias, accumulator domain
    input  logic signed [ACCW-1:0]   mult,    // fixed-point multiplier M
    input  logic [5:0]               shift,   // right-shift amount (0..63)
    input  logic                     relu,    // apply ReLU (clamp min to 0)
    output logic signed [OW-1:0]     y        // int8 activation for next layer
);
    localparam signed [OW-1:0] YMAX =  (1 <<< (OW-1)) - 1;   // +127
    localparam signed [OW-1:0] YMIN = -((1 <<< (OW-1)) - 1); // -127 (symmetric int8)

    logic signed [ACCW:0]     x;      // acc + bias  (one guard bit)
    logic signed [2*ACCW:0]   prod;   // x * mult    (wide enough: 33 + 32 bits)
    logic signed [2*ACCW:0]   half, r;

    always_comb begin
        x    = acc + bias;
        prod = x * mult;
        half = (shift != 6'd0) ? (({{(2*ACCW){1'b0}}, 1'b1}) <<< (shift - 6'd1)) : '0;
        r    = (prod + half) >>> shift;         // signed -> arithmetic shift
        if (relu && r < 0) r = '0;
        if      (r > YMAX) y = YMAX;
        else if (r < YMIN) y = YMIN;
        else               y = r[OW-1:0];
    end
endmodule
