"""Graded ('colour') spikes vs binary spikes — does a richer per-spike message help? (A/B)

Sergiy's idea: an SNN's weakness is a thin message (1 bit per spike), not fewer parameters.
Give each spike a 'colour' = how far past threshold the neuron was driven when it fired (a
confidence signal we currently discard). Test: same STDP feature net, same R-STDP readout,
change ONLY the message the readout sees.

  A (binary): feature vector = spike COUNTS  (what we do now)
  B (colour): feature vector = summed membrane overshoot per neuron (Loihi graded spike)

If B > A, the colour channel carried usable information the binary spike was throwing away.
Backprop-free throughout. Clean A/B: one trained net, two feature encodings, two readouts.

Usage:
    .venv/bin/python experiments/graded_spikes_ab.py --n-exc 800 --feat-train 30000 \
        --readout-train 20000 --n-test 10000
"""

import argparse
import time

import torch

from stdp_diehl_cook import STDPNet, load_mnist


def train_features(n_exc, X, T, passes, theta_plus, device, seed):
    net = STDPNet(784, n_exc, device, seed=seed, theta_plus=theta_plus)
    t0 = time.time()
    for p in range(passes):
        for i in range(X.size(0)):
            net.present_image(X[i], T, learn=True)
            if (i + 1) % 10000 == 0:
                print(f"  [features] pass {p+1} img {i+1}/{X.size(0)} ({time.time()-t0:.0f}s)")
    return net


def _l1(x):
    return x / x.sum(1, keepdim=True).clamp_min(1e-6)


def extract_all(net, X, T):
    """Four message encodings from one pass:
      counts   — binary spike counts (A)
      grad     — summed overshoot 'colour' (B)
      meanconf — mean overshoot per neuron = confidence DECOUPLED from count (C)
      concat   — [counts | meanconf]: does confidence ADD info beyond count? (D)"""
    n = X.size(0)
    counts = torch.zeros(n, net.n_exc, device=net.device)
    grad = torch.zeros(n, net.n_exc, device=net.device)
    meanconf = torch.zeros(n, net.n_exc, device=net.device)
    for i in range(n):
        c, g = net.present_image_graded(X[i], T)
        counts[i], grad[i] = c, g
        meanconf[i] = g / c.clamp_min(1e-6)          # avg overshoot where it fired, 0 if silent
    counts_n, grad_n, mc_n = _l1(counts), _l1(grad), _l1(meanconf)
    concat = torch.cat([counts_n, mc_n], dim=1)
    return {"counts": counts_n, "grad": grad_n, "meanconf": mc_n, "concat": concat}


def train_readout(feat, y, epochs, lr, device):
    W = torch.zeros(10, feat.size(1), device=device)
    for _ in range(epochs):
        for idx in torch.randperm(feat.size(0)).tolist():
            f, lab = feat[idx], int(y[idx])
            pred = int((W @ f).argmax())
            if pred != lab:
                W[lab] += lr * f
                W[pred] -= lr * f
    return W


def acc(W, feat, y):
    return (feat @ W.t()).argmax(1).eq(y).float().mean().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-exc", type=int, default=800)
    ap.add_argument("--feat-train", type=int, default=30000)
    ap.add_argument("--feat-passes", type=int, default=2)
    ap.add_argument("--readout-train", type=int, default=20000)
    ap.add_argument("--readout-epochs", type=int, default=15)
    ap.add_argument("--readout-lr", type=float, default=0.01)
    ap.add_argument("--n-test", type=int, default=10000)
    ap.add_argument("--T", type=int, default=100)
    ap.add_argument("--theta-plus", type=float, default=0.0005)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    print(f"device: {device} | n_exc: {args.n_exc} | binary vs graded A/B")

    n_load = max(args.feat_train, args.readout_train)
    Xtr, ytr, Xte, yte = load_mnist(n_load, args.n_test)
    ytr_r, yte = ytr[:args.readout_train].to(device), yte.to(device)

    print("\n[1] train STDP features (shared by both arms)...")
    net = train_features(args.n_exc, Xtr[:args.feat_train], args.T, args.feat_passes,
                         args.theta_plus, device, args.seed)

    print("[2] extract all four message encodings...")
    t0 = time.time()
    tr = extract_all(net, Xtr[:args.readout_train], args.T)
    te = extract_all(net, Xte, args.T)
    print(f"  extracted in {time.time()-t0:.0f}s")

    print("[3] train a readout on each encoding, same everything else...")
    results = {}
    for key in ("counts", "grad", "meanconf", "concat"):
        W = train_readout(tr[key], ytr_r, args.readout_epochs, args.readout_lr, device)
        results[key] = acc(W, te[key], yte)

    print("\n" + "=" * 60)
    print(f"  binary vs graded ('colour') spikes  |  {args.n_exc} neurons")
    print(f"  A  counts   (binary):                 {results['counts']*100:.2f}%")
    print(f"  B  grad     (summed colour):          {results['grad']*100:.2f}%")
    print(f"  C  meanconf (confidence, no count):   {results['meanconf']*100:.2f}%")
    print(f"  D  concat   [counts | confidence]:    {results['concat']*100:.2f}%")
    print(f"  decisive: D - A (does confidence ADD to count?): "
          f"{(results['concat']-results['counts'])*100:+.2f} pp")
    print("=" * 60)


if __name__ == "__main__":
    main()
