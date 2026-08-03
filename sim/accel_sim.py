#!/usr/bin/env python3
"""
Bit-exact functional model of the 8x8 int8 MAC accelerator, plus the DS-CNN layer
primitives built on top of it. This lets us run a whole keyword-spotting inference
"on the accelerator" -- using its exact int8/int32 arithmetic and its 8x8 tiling --
entirely in software, and check the result against a reference.

The core op mirrors rtl/mac_array_8x8.sv: each PE computes int8*int8 and accumulates
in a signed 32-bit register; one array pass turns an (<=8 x K) activation tile and a
(K x <=8) weight tile into an (<=8 x <=8) int32 output tile. Everything larger -- full
matmuls, and the convolutions lowered to matmuls (im2col) -- is tiled down to that
one op, exactly the way the control FSM sequences it on real hardware.

Run directly to self-test every DS-CNN layer type against NumPy references.
"""
import numpy as np

N = 8                                   # the array is N x N processing elements
_ACCMIN, _ACCMAX = -(2**31), 2**31 - 1  # 32-bit signed accumulator


def _sat32(x):
    return np.clip(x, _ACCMIN, _ACCMAX)


def tile_mac(A, W):
    """One array pass. A:(n<=8,K) int8, W:(K,m<=8) int8 -> O:(n,m) int32.
    Exactly what mac_array_8x8 computes: int8*int8 products summed in 32 bits."""
    A = np.asarray(A, np.int64)
    W = np.asarray(W, np.int64)
    assert A.shape[0] <= N and W.shape[1] <= N and A.shape[1] == W.shape[0]
    return _sat32(A @ W).astype(np.int32)


def matmul(A, W):
    """Arbitrary (M,K)x(K,P) int8 matmul, tiled into 8x8 array passes (int32 out)."""
    A = np.asarray(A, np.int64)
    W = np.asarray(W, np.int64)
    M, K = A.shape
    K2, P = W.shape
    assert K == K2, "inner dimensions must match"
    O = np.zeros((M, P), np.int64)
    for i in range(0, M, N):
        for j in range(0, P, N):
            O[i:i+N, j:j+N] = tile_mac(A[i:i+N, :], W[:, j:j+N])
    return _sat32(O).astype(np.int32)


# ---- convolution layers, lowered to matmul (NHWC, int8 in, int32 out) --------
def _im2col(x, R, S, stride, pad):
    """x:(H,W,C) -> columns:(P*Q, R*S*C) and output dims (P,Q)."""
    H, Wd, C = x.shape
    xp = np.pad(x, ((pad, pad), (pad, pad), (0, 0)))
    P = (H + 2*pad - R)//stride + 1
    Q = (Wd + 2*pad - S)//stride + 1
    cols = np.empty((P*Q, R*S*C), x.dtype)
    k = 0
    for p in range(P):
        for q in range(Q):
            patch = xp[p*stride:p*stride+R, q*stride:q*stride+S, :]
            cols[k] = patch.reshape(-1)
            k += 1
    return cols, P, Q


def conv2d(x, w, stride=1, pad=0):
    """Standard/pointwise conv on the accelerator. x:(H,W,Cin) int8,
    w:(R,S,Cin,Cout) int8 -> y:(P,Q,Cout) int32."""
    R, S, Cin, Cout = w.shape
    cols, P, Q = _im2col(x, R, S, stride, pad)          # (P*Q, R*S*Cin)
    wmat = w.reshape(R*S*Cin, Cout)                      # (R*S*Cin, Cout)
    y = matmul(cols, wmat)                               # tiled 8x8 matmuls
    return y.reshape(P, Q, Cout)


def depthwise2d(x, w, stride=1, pad=1):
    """Depthwise conv: one R*S filter per channel, no cross-channel mixing.
    x:(H,W,C) int8, w:(R,S,C) int8 -> y:(P,Q,C) int32. Each channel is an
    independent (P*Q x R*S)*(R*S x 1) matmul on the array."""
    R, S, C = w.shape
    outs = []
    for c in range(C):
        cols, P, Q = _im2col(x[:, :, c:c+1], R, S, stride, pad)   # (P*Q, R*S)
        wc = w[:, :, c].reshape(R*S, 1)                            # (R*S, 1)
        outs.append(matmul(cols, wc).reshape(P, Q, 1))
    return np.concatenate(outs, axis=2)


def fc(x_vec, w):
    """Fully-connected: x:(Cin,) int8, w:(Cin,Cout) int8 -> (Cout,) int32."""
    return matmul(np.asarray(x_vec, np.int64).reshape(1, -1), w).reshape(-1)


# ---- self-test: every layer type must match a NumPy reference exactly --------
def _selftest():
    rng = np.random.default_rng(0)
    i8 = lambda *s: rng.integers(-128, 128, size=s, dtype=np.int64)
    ok = True

    A, W = i8(20, 37), i8(37, 30)               # arbitrary matmul, odd sizes
    if not np.array_equal(matmul(A, W), (A @ W)):
        ok = False; print("FAIL matmul")

    x, w = i8(25, 5, 1), i8(10, 4, 1, 64)        # standard conv (DS-CNN conv1 shape)
    ref = np.tensordot(_im2col(x, 10, 4, 1, 0)[0].reshape(-1, 40), w.reshape(40, 64), 1)
    if not np.array_equal(conv2d(x, w).reshape(-1, 64), ref):
        ok = False; print("FAIL conv2d")

    x, w = i8(25, 5, 64), i8(1, 1, 64, 64)       # pointwise 1x1
    if not np.array_equal(conv2d(x, w),
                          np.tensordot(x, w[0, 0], axes=([2], [0]))):
        ok = False; print("FAIL pointwise")

    x, w = i8(25, 5, 64), i8(3, 3, 64)           # depthwise 3x3, pad 1
    y = depthwise2d(x, w, stride=1, pad=1)
    xp = np.pad(x, ((1, 1), (1, 1), (0, 0)))
    ref = np.zeros_like(y)
    for p in range(25):
        for q in range(5):
            ref[p, q] = (xp[p:p+3, q:q+3, :] * w).sum(axis=(0, 1))
    if not np.array_equal(y, ref):
        ok = False; print("FAIL depthwise")

    v, w = i8(64), i8(64, 12)                    # FC 64->12
    if not np.array_equal(fc(v, w), (v @ w)):
        ok = False; print("FAIL fc")

    print("PASS: accelerator model is bit-exact for matmul, conv, pointwise, "
          "depthwise, and FC" if ok else "FAIL: see above")
    return ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if _selftest() else 1)
