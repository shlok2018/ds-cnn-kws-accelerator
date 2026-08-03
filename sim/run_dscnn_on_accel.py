#!/usr/bin/env python3
"""
Run a full DS-CNN keyword-spotting inference *on the accelerator*.

Every convolution and the final FC are executed through accel_sim -- i.e. the
8x8 int8 MAC array's exact arithmetic and tiling -- on real MFCC clips from the
MLPerf Tiny test set. BatchNorm is folded into the preceding conv; activations
and weights are quantized to int8 (per-clip dynamic activation scale, per-output-
channel weight scale); ReLU and the global average pool run in the surrounding
float logic (as they would in the accelerator's activation unit).

Inputs (produced earlier in the repo):
  kws_eval/dscnn_weights.npz   (from kws_eval/extract_weights.py, needs TF once)
  kws_eval/test_data.npz       (MFCC test clips + labels, from eval_precision.py)

Reports the accelerator's top-1 accuracy and a few example clip -> keyword calls.
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import accel_sim as acc

HERE = os.path.dirname(os.path.abspath(__file__))
KWS  = os.path.join(HERE, "..", "kws_eval")
LABELS = ["down","go","left","no","off","on","right","stop","up","yes",
          "silence","unknown"]           # order per the reference; adjust if needed

EPS = 1e-3                                # BatchNorm epsilon (Keras default)


def q_int8(x, axis=None):
    """Symmetric int8 quantize. Returns (int8 array, scale)."""
    m = np.max(np.abs(x), axis=axis, keepdims=True)
    m = np.where(m == 0, 1e-9, m)
    scale = m / 127.0
    q = np.clip(np.round(x / scale), -127, 127).astype(np.int8)
    return q, scale


def same_pad(x, R, S, stride):
    """Apply TensorFlow 'SAME' padding (asymmetric) and return padded x."""
    H, W = x.shape[:2]
    oH, oW = -(-H // stride), -(-W // stride)          # ceil
    pH = max((oH - 1) * stride + R - H, 0)
    pW = max((oW - 1) * stride + S - W, 0)
    t, b = pH // 2, pH - pH // 2
    l, r = pW // 2, pW - pW // 2
    return np.pad(x, ((t, b), (l, r), (0, 0)))


def fold_bn(conv_bias, gamma, beta, mean, var):
    """Fold conv bias + BatchNorm into per-channel (scale, bias):
    out = s * conv(x, W) + bias, so folded W = W*s. Handles use_bias=True convs."""
    s = gamma / np.sqrt(var + EPS)
    return s, (beta + s * (conv_bias - mean))


def conv_layer(x, kernel, conv_bias, bn, stride, kind):
    """kind: 'std' | 'pw' | 'dw'. Fold BN, quantize, run on the accelerator, ReLU."""
    if kind == "dw":                        # (R,S,C,1) -> (R,S,C)
        R, S, C, _ = kernel.shape
        k = kernel[:, :, :, 0]
        s, bias = fold_bn(conv_bias, *bn)   # per channel C
        kf = k * s.reshape(1, 1, C)
        xq, xs = q_int8(x)                  # per-tensor activation scale
        wq, ws = q_int8(kf, axis=(0, 1))    # per-channel weight scale (1,1,C)
        xp = same_pad(xq, R, S, stride)
        acc32 = acc.depthwise2d(xp, wq, stride=stride, pad=0)     # int32 (P,Q,C)
        y = acc32.astype(np.float64) * (xs * ws.reshape(1, 1, C)) + bias.reshape(1, 1, C)
    else:                                   # standard / pointwise (R,S,Cin,Cout)
        R, S, Cin, Cout = kernel.shape
        s, bias = fold_bn(conv_bias, *bn)   # per output channel Cout
        kf = kernel * s.reshape(1, 1, 1, Cout)
        xq, xs = q_int8(x)
        wq, ws = q_int8(kf.reshape(R*S*Cin, Cout), axis=0)        # per-out-channel
        wq = wq.reshape(R, S, Cin, Cout)
        xp = same_pad(xq, R, S, stride)
        acc32 = acc.conv2d(xp, wq, stride=stride, pad=0)          # int32 (P,Q,Cout)
        y = acc32.astype(np.float64) * (xs * ws.reshape(1, 1, Cout)) + bias.reshape(1, 1, Cout)
    return np.maximum(y, 0.0)               # ReLU


def load_weights():
    d = np.load(os.path.join(KWS, "dscnn_weights.npz"))
    # group arrays by layer index Lxx, keep class name and ordered weight list
    layers = {}
    for k in d.files:
        _, cls, j = k.split("_", 2) if k.count("_") >= 2 else (k, "", "0")
        idx = int(k[1:3]); cls = k.split("_")[1]
        layers.setdefault(idx, {"cls": cls, "w": {}})["w"][int(k.split("_")[-1])] = d[k]
    seq = []
    for idx in sorted(layers):
        L = layers[idx]
        seq.append((L["cls"], [L["w"][j] for j in sorted(L["w"])]))
    return seq


def infer(x, seq):
    """x: (49,10,1) float MFCC -> 12 logits, running convs/FC on the accelerator."""
    a = x.astype(np.float64)
    i, first_conv = 0, True
    while i < len(seq):
        cls, w = seq[i]
        if cls in ("Conv2D", "DepthwiseConv2D"):
            bn = seq[i+1][1] if i+1 < len(seq) and seq[i+1][0] == "BatchNormalization" else None
            if cls == "DepthwiseConv2D":
                a = conv_layer(a, w[0], w[1], bn, 1, "dw")
            else:
                R, S = w[0].shape[:2]
                kind = "pw" if (R == 1 and S == 1) else "std"
                a = conv_layer(a, w[0], w[1], bn, 2 if first_conv else 1, kind)
                first_conv = False
            i += 2 if bn else 1
        elif cls == "Dense":
            v = a.mean(axis=(0, 1))                       # global average pool -> (C,)
            vq, vs = q_int8(v)
            wq, ws = q_int8(w[0], axis=0)                 # (Cin,Cout)
            logits = acc.fc(vq, wq).astype(np.float64) * (vs * ws) + (w[1] if len(w) > 1 else 0)
            return logits
        else:
            i += 1
    raise RuntimeError("no Dense layer found")


def main():
    seq = load_weights()
    print("=== layer pipeline ===")
    for cls, w in seq:
        print(f"  {cls:22s} {[list(a.shape) for a in w]}")
    d = np.load(os.path.join(KWS, "test_data.npz"))
    X, y = d["X"], d["y"]
    n = int(os.environ.get("N_CLIPS", "500"))
    idx = np.arange(min(n, len(X)))
    preds = np.array([int(np.argmax(infer(X[k], seq))) for k in idx])
    acc_top1 = float((preds == y[idx]).mean())
    print(f"\naccelerator top-1 accuracy on {len(idx)} clips: {acc_top1*100:.2f}%")
    print("example calls (clip -> predicted keyword | true):")
    for k in idx[:8]:
        p = int(np.argmax(infer(X[k], seq)))
        print(f"  clip {k:4d} -> {LABELS[p]:8s} | true {LABELS[int(y[k])]}")


if __name__ == "__main__":
    main()
