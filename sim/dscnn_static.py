#!/usr/bin/env python3
"""
Static-quantized DS-CNN inference that matches the RTL sequencer (rtl/dscnn_seq)
exactly, and the calibration that produces it.

run_dscnn_on_accel.py uses *dynamic* per-clip activation scales -- fine in
software, impossible as a fixed hardware descriptor. Here we instead:
  1. calibrate one *static* per-tensor activation scale per layer (from data),
  2. fold each layer to integer requant params (mult, shift, bias) that combine
     the input scale, per-output-channel weight scale, and BN bias, exactly the
     r = clip(relu(round((acc+bias)*mult/2^shift)), int8) the hardware computes,
  3. run the whole network in integer arithmetic (accel_sim), and
  4. report accuracy.
This is the software golden the RTL is checked bit-exact against, and it emits the
descriptor/weight/param tables the hardware loads.
"""
import os, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import accel_sim as acc
from run_dscnn_on_accel import load_weights, same_pad, fold_bn, EPS, LABELS

KWS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "kws_eval")
SH_FC = 24                       # common fixed-point shift for the FC output logits


# ---- build the folded layer list: conv1, [dw, pw]*4, fc --------------------
def build_layers(seq):
    layers, i, first = [], 0, True
    while i < len(seq):
        cls, w = seq[i]
        if cls in ("Conv2D", "DepthwiseConv2D"):
            bn = seq[i+1][1] if i+1 < len(seq) and seq[i+1][0] == "BatchNormalization" else None
            if cls == "DepthwiseConv2D":
                R, S, C, _ = w[0].shape
                s, bias = fold_bn(w[1], *bn)
                kf = w[0][:, :, :, 0] * s.reshape(1, 1, C)         # (R,S,C)
                layers.append(("dw", kf, bias, 1))
            else:
                R, S, Ci, Co = w[0].shape
                s, bias = fold_bn(w[1], *bn)
                kf = w[0] * s.reshape(1, 1, 1, Co)                  # (R,S,Ci,Co)
                layers.append(("pw" if (R == 1 and S == 1) else "std", kf, bias, 2 if first else 1))
                first = False
            i += 2 if bn else 1
        elif cls == "Dense":
            layers.append(("fc", w[0], w[1] if len(w) > 1 else np.zeros(w[0].shape[1])))
            i += 1
        else:
            i += 1
    return layers


def conv_float(a, kf, kind, stride):
    """Exact float conv (for calibration ranges), matching accel_sim's lowering."""
    if kind == "dw":
        R, S, C = kf.shape
        xp = same_pad(a, R, S, stride)
        outs = []
        for c in range(C):
            cols, P, Q = acc._im2col(xp[:, :, c:c+1], R, S, stride, 0)
            outs.append((cols @ kf[:, :, c].reshape(R*S, 1)).reshape(P, Q, 1))
        return np.concatenate(outs, axis=2)
    R, S, Ci, Co = kf.shape
    xp = same_pad(a, R, S, stride)
    cols, P, Q = acc._im2col(xp, R, S, stride, 0)
    return (cols @ kf.reshape(R*S*Ci, Co)).reshape(P, Q, Co)


# ---- calibration: max |activation| entering each layer, over a few clips ----
def calibrate(X, layers, n):
    nL = len(layers)
    amax = np.zeros(nL)      # max |input| to layer li
    fcmax = 0.0
    for idx in range(min(n, len(X))):
        a = X[idx].astype(np.float64)
        for li, L in enumerate(layers):
            amax[li] = max(amax[li], np.max(np.abs(a)))
            if L[0] == "fc":
                v = a.mean(axis=(0, 1))
                fcmax = max(fcmax, np.max(np.abs(v)))
                break
            a = np.maximum(conv_float(a, L[1], L[0], L[3]) + L[2].reshape(1, 1, -1), 0.0)
    s = amax / 127.0                       # per-layer input scale (index li = input to layer li)
    return s, fcmax / 127.0


# ---- derive integer weights + requant params from the float layers ---------
def q_w_perout(kf, kind):
    """Per-output-channel symmetric int8 weights + scale."""
    if kind == "dw":
        R, S, C = kf.shape
        ws = np.maximum(np.max(np.abs(kf.reshape(R*S, C)), axis=0), 1e-12) / 127.0
        wq = np.clip(np.round(kf / ws.reshape(1, 1, C)), -127, 127).astype(np.int8)
        return wq, ws
    R, S, Ci, Co = kf.shape
    flat = kf.reshape(R*S*Ci, Co)
    ws = np.maximum(np.max(np.abs(flat), axis=0), 1e-12) / 127.0
    wq = np.clip(np.round(flat / ws.reshape(1, Co)), -127, 127).astype(np.int8).reshape(R, S, Ci, Co)
    return wq, ws


def mult_shift(M):
    """Represent positive floats M[.] as (mult, shift) with a common shift so the
    largest multiplier lands near 2^30 (max int32 precision)."""
    mmax = float(np.max(M))
    shift = int(np.clip(30 - np.floor(np.log2(mmax + 1e-30)), 0, 40))
    mult = np.round(M * (2.0 ** shift)).astype(np.int64)
    return mult.astype(np.int32), shift


def quantize(layers, s, s_fc):
    q = []
    for li, L in enumerate(layers):
        if L[0] == "fc":
            W, b = L[1], L[2]                         # (Cin,Cout), (Cout,)
            fcws = np.maximum(np.max(np.abs(W), axis=0), 1e-12) / 127.0
            fcwq = np.clip(np.round(W / fcws.reshape(1, -1)), -127, 127).astype(np.int8)
            fc_mult = np.round(s_fc * fcws * (2.0 ** SH_FC)).astype(np.int64)
            fc_bias = np.round(b * (2.0 ** SH_FC)).astype(np.int64)
            q.append(("fc", fcwq, fc_mult, fc_bias))
        else:
            kind, kf, bias, stride = L
            wq, ws = q_w_perout(kf, kind)
            s_in, s_out = s[li], s[li+1]
            M = s_in * ws / s_out                     # per output channel
            mult, shift = mult_shift(M)
            bias_i = np.round(bias / (s_in * ws)).astype(np.int32)
            q.append((kind, wq, mult, shift, bias_i, stride))
    return q


# ---- static integer inference (mirrors the hardware) -----------------------
def infer_static(x, q, s0, s_fc, s_last, npos):
    a = np.clip(np.round(x / s0), -127, 127).astype(np.int8)     # quantize MFCC
    for L in q:
        if L[0] == "fc":
            _, fcwq, fc_mult, fc_bias = L
            vsum = a.reshape(-1, a.shape[-1]).sum(axis=0).astype(np.int64)      # pool: sum positions
            pool_M = s_last / (npos * s_fc)
            pm, psh = mult_shift(np.array([pool_M]))
            vq = acc.requant(vsum, int(pm[0]), int(psh), 0, relu=True)          # -> int8
            raw = acc.fc(vq, fcwq).astype(np.int64)
            return int(np.argmax(raw * fc_mult + fc_bias))
        kind, wq, mult, shift, bias_i, stride = L
        xp = same_pad(a, wq.shape[0], wq.shape[1], stride)
        acc32 = acc.depthwise2d(xp, wq, stride, 0) if kind == "dw" else acc.conv2d(xp, wq, stride, 0)
        a = acc.requant(acc32, mult, shift, bias_i, relu=True)
    raise RuntimeError("no fc")


def same_pad_amt(H, W, R, S, stride):
    P, Q = -(-H // stride), -(-W // stride)
    pH = max((P-1)*stride + R - H, 0); pW = max((Q-1)*stride + S - W, 0)
    return P, Q, pH // 2, pW // 2


def emit_rtl(X, y, layers, q, s, s_fc, npos, nclips, outdir):
    """Write the descriptor/weight/param tables + a few quantized clips for the
    RTL sequencer (rtl/dscnn_seq) to read via $readmemh, plus expected preds."""
    os.makedirs(outdir, exist_ok=True)
    NL = len(layers)
    h8  = lambda v: format(int(v) & 0xFF, "02x")
    h32 = lambda v: format(int(v) & 0xFFFFFFFF, "08x")
    desc = [[0]*17 for _ in range(NL)]
    wmem, pmult, pshift, pbias = [], [], [], []
    H, W, C = X[0].shape                                   # 49,10,1 input map
    for li, (L, ql) in enumerate(zip(layers, q)):
        woff, poff = len(wmem), len(pmult)
        if ql[0] == "fc":
            _, fcwq, fc_mult, fc_bias = ql
            Cin, NC = fcwq.shape
            for c in range(Cin):
                for o in range(NC): wmem.append(int(fcwq[c, o]))
            pool_M = s[li] / (npos * s_fc)
            pm, psh = mult_shift(np.array([pool_M]))
            for o in range(NC):
                pmult.append(int(fc_mult[o])); pshift.append(0); pbias.append(int(fc_bias[o]))
            # H,W of the fc "layer" = input spatial (so M=H*W); Cout=NC
            desc[li] = [H, W, C, 1, 1, 1, NC, 0, 0, 0, 0, 0, woff, poff, 2, int(pm[0]), int(psh)]
        elif ql[0] == "dw":
            _, wq, mult, shift, bias_i, stride = ql
            R, S, Cc = wq.shape
            P, Q, pt, pl = same_pad_amt(H, W, R, S, stride)
            for ci in range(Cc):
                for k in range(R*S): wmem.append(int(wq[k//S, k % S, ci]))
            for ci in range(Cc):
                pmult.append(int(mult[ci])); pshift.append(int(shift)); pbias.append(int(bias_i[ci]))
            desc[li] = [H, W, C, R, S, stride, Cc, P, Q, pt, pl, 1, woff, poff, 1, 0, 0]
            H, W, C = P, Q, Cc
        else:                                             # std / pw
            _, wq, mult, shift, bias_i, stride = ql
            R, S, Ci, Co = wq.shape
            P, Q, pt, pl = same_pad_amt(H, W, R, S, stride)
            flat = wq.reshape(R*S*Ci, Co)
            for k in range(R*S*Ci):
                for o in range(Co): wmem.append(int(flat[k, o]))
            for o in range(Co):
                pmult.append(int(mult[o])); pshift.append(int(shift)); pbias.append(int(bias_i[o]))
            desc[li] = [H, W, C, R, S, stride, Co, P, Q, pt, pl, 1, woff, poff, 0, 0, 0]
            H, W, C = P, Q, Co
    with open(f"{outdir}/desc.hex", "w") as f:
        for l in range(NL):
            for fld in range(17): f.write(h32(desc[l][fld]) + "\n")
    with open(f"{outdir}/wmem.hex", "w") as f:  f.write("\n".join(h8(v) for v in wmem) + "\n")
    with open(f"{outdir}/pmult.hex", "w") as f: f.write("\n".join(h32(v) for v in pmult) + "\n")
    with open(f"{outdir}/pshift.hex", "w") as f: f.write("\n".join(h32(v) for v in pshift) + "\n")
    with open(f"{outdir}/pbias.hex", "w") as f: f.write("\n".join(h32(v) for v in pbias) + "\n")
    s_last = s[NL-1]
    clips, preds = [], []
    for k in range(nclips):
        xq = np.clip(np.round(X[k] / s[0]), -127, 127).astype(np.int8).reshape(-1)
        clips += [int(v) for v in xq]
        preds.append(infer_static(X[k], q, s[0], s_fc, s_last, npos))
    with open(f"{outdir}/clips.hex", "w") as f:  f.write("\n".join(h8(v) for v in clips) + "\n")
    with open(f"{outdir}/preds.hex", "w") as f:  f.write("\n".join(format(p, "x") for p in preds) + "\n")
    with open(f"{outdir}/meta.txt", "w") as f:
        f.write(f"{NL} {len(wmem)} {len(pmult)} {nclips} {H*W*C if False else 490}\n")
    print(f"emitted {NL} layers, {len(wmem)} weights, {len(pmult)} params, {nclips} clips -> {outdir}")
    print(f"  expected preds: {preds}")


def main():
    seq = load_weights()
    layers = build_layers(seq)
    d = np.load(os.path.join(KWS, "test_data.npz"))
    X, y = d["X"], d["y"]
    ncal = int(os.environ.get("N_CALIB", "100"))
    ntest = int(os.environ.get("N_CLIPS", "500"))

    s, s_fc = calibrate(X, layers, ncal)
    q = quantize(layers, s, s_fc)
    s_last = s[len(layers)-1]                    # input scale of the fc layer = last conv output scale
    npos = None
    # one warm inference to learn npos (P*Q of the last conv output)
    a = np.clip(np.round(X[0] / s[0]), -127, 127).astype(np.int8)
    for L in q[:-1]:
        kind, wq, mult, shift, bias_i, stride = L
        xp = same_pad(a, wq.shape[0], wq.shape[1], stride)
        acc32 = acc.depthwise2d(xp, wq, stride, 0) if kind == "dw" else acc.conv2d(xp, wq, stride, 0)
        a = acc.requant(acc32, mult, shift, bias_i, relu=True)
    npos = a.shape[0] * a.shape[1]

    idx = np.arange(min(ntest, len(X)))
    preds = np.array([infer_static(X[k], q, s[0], s_fc, s_last, npos) for k in idx])
    acc_top1 = float((preds == y[idx]).mean())
    print(f"layers: {[L[0] for L in layers]}")
    print(f"calibrated scales (per-layer input): {np.array2string(s, precision=4)}")
    print(f"static-quant accelerator top-1 accuracy on {len(idx)} clips: {acc_top1*100:.2f}%")

    if os.environ.get("EMIT"):
        n = int(os.environ.get("EMIT", "5"))
        here = os.path.dirname(os.path.abspath(__file__))
        emit_rtl(X, y, layers, q, s, s_fc, npos, n, os.path.join(here, "..", "rtl", "gen"))


if __name__ == "__main__":
    main()
