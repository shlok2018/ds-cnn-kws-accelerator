#!/usr/bin/env python3
# ============================================================================
# make_problems.py
# Emits Timeloop v4 "problem" YAMLs for the three layer TYPES in a DS-CNN.
#
# The whole point of splitting these out: a depthwise-separable CNN is really
# two very different computations stapled together, and they stress a systolic
# array in OPPOSITE ways. If you feed Timeloop the averaged whole-network stats
# you'll never see it. Split them and the story jumps out:
#
#   standard conv  -> lots of MACs, lots of reuse  -> array stays busy  (good)
#   depthwise      -> one filter per channel, ZERO cross-channel reuse
#                     -> a big array sits mostly IDLE                    (bad!)
#   pointwise 1x1  -> pure channel mixing, huge reuse -> array busy     (good)
#
# The dimension names are Timeloop's CNN convention:
#   N = batch        (1 for edge inference)
#   M = output channels
#   C = input channels
#   P = output height,  Q = output width
#   R = filter height,  S = filter width
#
# Shapes below are representative of the MLPerf Tiny DS-CNN keyword-spotter
# operating on a 49x10 MFCC "image". Tweak to match your exact trained model;
# the RELATIVE behaviour between the three is what matters, not exact sizes.
# ============================================================================

import os
import textwrap

OUT = os.path.join(os.path.dirname(__file__), "..", "problems")
os.makedirs(OUT, exist_ok=True)


def problem_yaml(name, N, M, C, P, Q, R, S, note):
    """Build one Timeloop v4 problem spec."""
    return textwrap.dedent(f"""\
        # {name}  --  {note}
        problem:
          version: 0.4
          instance:
            N: {N}
            M: {M}
            C: {C}
            P: {P}
            Q: {Q}
            R: {R}
            S: {S}
          shape:
            name: "CNN_Layer"
            dimensions: [C, M, R, S, N, P, Q]
            data_spaces:
            - name: Weights
              projection:
              - [[C]]
              - [[M]]
              - [[R]]
              - [[S]]
            - name: Inputs
              projection:
              - [[N]]
              - [[C]]
              - [[R], [P]]     # input row = filter row + output row (conv)
              - [[S], [Q]]     # input col = filter col + output col
            - name: Outputs
              projection:
              - [[N]]
              - [[M]]
              - [[P]]
              - [[Q]]
              read_write: True
        """)


def depthwise_yaml(name, N, G, P, Q, R, S, note):
    """Build a TRUE depthwise (grouped-conv) Timeloop v4 problem.

    The group dimension G (== number of channels) gives one independent filter
    per channel with C=1, M=1 inside each group -> ZERO cross-channel reduction.
    MACs = G*C*M*R*S*P*Q = G*R*S*P*Q, not the M*C*... of a full conv. C and M are
    kept (both 1) so the shared weight-stationary constraints still resolve.
    """
    return textwrap.dedent(f"""\
        # {name}  --  {note}
        problem:
          version: 0.4
          instance:
            N: {N}
            G: {G}
            C: 1
            M: 1
            P: {P}
            Q: {Q}
            R: {R}
            S: {S}
          shape:
            name: "DepthwiseConv"
            dimensions: [G, C, M, R, S, N, P, Q]
            data_spaces:
            - name: Weights
              projection:
              - [[G]]          # one filter per group/channel
              - [[C]]
              - [[M]]
              - [[R]]
              - [[S]]
            - name: Inputs
              projection:
              - [[N]]
              - [[G]]          # input channel = group (C=1 within it)
              - [[C]]
              - [[R], [P]]
              - [[S], [Q]]
            - name: Outputs
              projection:
              - [[N]]
              - [[G]]          # output channel = SAME group -> depthwise tie
              - [[M]]
              - [[P]]
              - [[Q]]
              read_write: True
        """)


# ---- Layer shapes: grounded in the MLPerf Tiny KWS DS-CNN reference --------
# The reference model (input 49x10 MFCC, 12-class softmax) is:
#   conv1 (standard) -> 4x [depthwise 3x3 + pointwise 1x1] -> avgpool -> FC(->12)
# So a full inference runs 1 standard, 4 depthwise, 4 pointwise, 1 FC (avgpool
# has no MACs). Those counts become the layer weights in analyze.py.
#
# NOTE (documented approximation): conv1 is stride-2 in the real model. We model
# it stride-1 here -- the OUTPUT dims (P=25,Q=5) and MAC count are already
# correct, so only conv1's input footprint is slightly understated. conv1 is 1
# of 9 conv layers and the depthwise/pointwise blocks dominate, so the effect on
# the per-inference total is negligible. Refine with stride coefficients if you
# need conv1's roofline point to be exact.

# Standard + pointwise + FC are ordinary (full) convs -> problem_yaml.
full_convs = [
    # conv1: 10x4 filter, 1->64 channels. Healthy reuse.
    dict(name="conv_standard", N=1, M=64, C=1,  P=25, Q=5, R=10, S=4,
         note="standard conv (first layer, full filter) -- array stays busy"),

    # Pointwise 1x1: pure 1x1 conv mixing all channels. R=S=1, big C and M.
    # Maps beautifully onto a systolic array.
    dict(name="conv_pointwise", N=1, M=64, C=64, P=25, Q=5, R=1, S=1,
         note="pointwise 1x1 -- pure channel mixing, array stays busy"),

    # Final classifier: FC(64->12) after global avg-pool, modeled as a 1x1 conv
    # over a 1x1 spatial map. Tiny (768 MACs) but part of a real inference.
    dict(name="conv_fc", N=1, M=12, C=64, P=1, Q=1, R=1, S=1,
         note="FC classifier 64->12 (post avgpool) -- negligible compute"),
]

# Depthwise is a TRUE grouped conv (G groups of 1) -> depthwise_yaml. Each of the
# 64 channels gets its own 3x3 filter with no cross-channel reduction, so MACs
# are ~64x fewer than the old full-conv model and the reduction axis is starved.
depthwise = dict(name="conv_depthwise", N=1, G=64, P=25, Q=5, R=3, S=3,
                 note="TRUE depthwise 3x3 (grouped, G=64) -- no cross-channel reduction")

for L in full_convs:
    path = os.path.join(OUT, L["name"] + ".yaml")
    with open(path, "w") as f:
        f.write(problem_yaml(**L))
    print(f"wrote {path}")

path = os.path.join(OUT, depthwise["name"] + ".yaml")
with open(path, "w") as f:
    f.write(depthwise_yaml(**depthwise))
print(f"wrote {path}")

print("\nNOTE: depthwise is now a real grouped conv (G=64 groups of C=1,M=1), so")
print("its MACs = G*R*S*P*Q, not M*C*R*S*P*Q. The old model over-counted ~64x.")
