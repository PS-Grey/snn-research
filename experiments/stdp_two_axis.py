"""Idea 3: does the graded 'colour' channel carry a second, orthogonal axis? (two-axis test)

Setup measurement showed count and per-neuron overshoot are nearly independent channels
(r=0.045). So on MNIST (single axis = digit) they were redundant only because both encode the
one available label. This tests a genuinely TWO-axis task: each image has a digit (axis 1) AND an
independent contrast level (axis 2, a magnitude property that should live in firing STRENGTH =
overshoot, not in which neurons fire). Ramping OFF so contrast is not normalised away.

Question: on the CONTRAST head, do graded features (count + overshoot) beat binary count? If yes,
the overshoot 'colour' channel carries the second axis that binary spikes discard — Sergiy's
original idea, on data where it is not redundant by construction.

Usage:
    .venv/bin/python experiments/stdp_two_axis.py --n-exc 800 --feat-train 20000 --n-test 10000
"""

import argparse
import time

import torch
import torch.nn.functional as F

from stdp_diehl_cook import STDPNet, load_mnist

DELTA_GAIN = 10.0
CONTRAST = torch.tensor([0.7, 1.0, 1.4])   # three independent contrast levels (axis 2)


def apply_contrast(X, c):
    return (X * CONTRAST[c].view(-1, 1)).clamp(0, 1)


def train_unsup(net, X, T, passes):
    t0 = time.time()
    for p in range(passes):
        for i in range(X.size(0)):
            net.present_image(X[i], T, learn=True, min_spikes=0)   # min_spikes=0 -> no ramping
            if (i + 1) % 10000 == 0:
                print(f"  [unsup] pass {p+1} img {i+1}/{X.size(0)} ({time.time()-t0:.0f}s)")


def _l1(x):
    return x / x.sum(1, keepdim=True).clamp_min(1e-6)


def extract(net, X, T):
    """binary = counts; graded = [counts | mean-overshoot] (no ramping)."""
    n = X.size(0)
    C = torch.zeros(n, net.n_exc); G = torch.zeros(n, net.n_exc)
    for i in range(n):
        c, g = net.present_image_graded(X[i], T, min_spikes=0)
        C[i] = c; G[i] = g
    counts = _l1(C)
    meanconf = _l1(G / C.clamp_min(1e-6))
    return counts, torch.cat([counts, meanconf], dim=1)


def train_readout(feat, y, n_cls, epochs, lr):
    W = torch.zeros(n_cls, feat.size(1))
    for _ in range(epochs):
        for idx in torch.randperm(feat.size(0)).tolist():
            f, lab = feat[idx], int(y[idx])
            p = F.softmax(W @ f, dim=0)
            t = torch.zeros(n_cls); t[lab] = 1.0
            W += DELTA_GAIN * lr * (t - p).unsqueeze(1) * f.unsqueeze(0)
    return W


def acc(W, feat, y):
    return (feat @ W.t()).argmax(1).eq(y).float().mean().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-exc", type=int, default=800)
    ap.add_argument("--feat-train", type=int, default=20000)
    ap.add_argument("--feat-passes", type=int, default=2)
    ap.add_argument("--readout-train", type=int, default=15000)
    ap.add_argument("--readout-epochs", type=int, default=15)
    ap.add_argument("--readout-lr", type=float, default=0.05)
    ap.add_argument("--n-test", type=int, default=10000)
    ap.add_argument("--T", type=int, default=100)
    ap.add_argument("--theta-plus", type=float, default=0.0005)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    dev = torch.device("cpu")
    torch.manual_seed(args.seed)
    print(f"n_exc: {args.n_exc} | two-axis (digit + contrast) | binary vs graded")

    n_load = max(args.feat_train, args.readout_train)
    Xtr, ytr, Xte, yte = load_mnist(n_load, args.n_test)
    # axis 2: independent random contrast label per image
    ctr = torch.randint(0, 3, (n_load,)); cte = torch.randint(0, 3, (args.n_test,))
    Xtr_c = apply_contrast(Xtr, ctr); Xte_c = apply_contrast(Xte, cte)

    print("\n[1] unsupervised STDP features on contrast-varied images...")
    net = STDPNet(784, args.n_exc, dev, seed=args.seed, theta_plus=args.theta_plus)
    train_unsup(net, Xtr_c[:args.feat_train], args.T, args.feat_passes)

    print("[2] extract binary + graded features...")
    bin_tr, grad_tr = extract(net, Xtr_c[:args.readout_train], args.T)
    bin_te, grad_te = extract(net, Xte_c, args.T)

    print("[3] two heads (digit, contrast) x two encodings (binary, graded)...")
    heads = {"digit": (ytr[:args.readout_train], yte, 10),
             "contrast": (ctr[:args.readout_train], cte, 3)}
    res = {}
    for name, (ytr_h, yte_h, ncls) in heads.items():
        for enc, (ftr, fte) in (("binary", (bin_tr, bin_te)), ("graded", (grad_tr, grad_te))):
            W = train_readout(ftr, ytr_h, ncls, args.readout_epochs, args.readout_lr)
            res[(name, enc)] = acc(W, fte, yte_h)

    print("\n" + "=" * 60)
    print(f"  two-axis: does graded carry the orthogonal (contrast) axis?  {args.n_exc} neurons")
    for name in ("digit", "contrast"):
        b, g = res[(name, "binary")], res[(name, "graded")]
        print(f"  {name:9s}  binary {b*100:5.2f}%   graded {g*100:5.2f}%   Δ {(g-b)*100:+.2f} pp")
    print("  (chance: digit 10%, contrast 33%)")
    print("=" * 60)


if __name__ == "__main__":
    main()
