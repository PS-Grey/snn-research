"""Confidence feeds LEARNING — surprise-weighted vs confidence-weighted readout (three-factor).

The graded-spike test showed confidence doesn't help as a readout *feature* (redundant with
count). This tests the other mechanism: confidence feeding the *learning rule*. Sergiy's concern:
"learn more from confident spikes" overfloods with easy cases and ignores the hard/uncertain
ones ("not punishing more complex guesses"). The fix is to learn from ERROR/SURPRISE, not
confidence — the delta rule / focal-loss / prediction-error principle (still local, no backprop).

Three readout learning rules on the same frozen STDP features (arms share features/data/seed):
  A flat        — perceptron: on error, W[label]+=lr*f, W[pred]-=lr*f (our current rule)
  B surprise    — delta rule: W += lr*(onehot - softmax(scores)) ⊗ f  (learn from prediction
                  error; confident-correct → ~0 update, confident-wrong → big update)
  C confidence  — naive "learn from confident": B scaled by max(p) → amplifies confident,
                  ignores uncertain cases. Should underperform (demonstrates Sergiy's concern).

Applied to the READOUT only (not the features): the recorded 'spike-rate as difficulty signal ->
unstable' failure mode is a circular feedback into the *representation*; the readout has no such
loop, so this is the safe place to test surprise-gated learning.

Usage:
    .venv/bin/python experiments/stdp_surprise_learning.py --n-exc 800 --feat-train 30000 \
        --readout-train 20000 --n-test 10000 --readout-epochs 15
"""

import argparse
import time

import torch
import torch.nn.functional as F

from stdp_diehl_cook import STDPNet, load_mnist

DELTA_GAIN = 10.0   # matches the delta-rule effective learning rate to the perceptron's kicks


def train_features(n_exc, X, T, passes, theta_plus, device, seed):
    net = STDPNet(784, n_exc, device, seed=seed, theta_plus=theta_plus)
    t0 = time.time()
    for p in range(passes):
        for i in range(X.size(0)):
            net.present_image(X[i], T, learn=True)
            if (i + 1) % 10000 == 0:
                print(f"  [features] pass {p+1} img {i+1}/{X.size(0)} ({time.time()-t0:.0f}s)")
    return net


def extract(net, X, T):
    feats = torch.zeros(X.size(0), net.n_exc, device=net.device)
    for i in range(X.size(0)):
        feats[i] = net.present_image(X[i], T, learn=False)
    return feats / feats.sum(1, keepdim=True).clamp_min(1e-6)


def train_rule(rule, feat, y, feat_te, y_te, epochs, lr, device):
    """Returns (final test acc, per-epoch test-acc trace) for one learning rule."""
    W = torch.zeros(10, feat.size(1), device=device)
    N = feat.size(0)
    trace = []
    for _ in range(epochs):
        for idx in torch.randperm(N).tolist():
            f, lab = feat[idx], int(y[idx])
            s = W @ f
            if rule == "flat":
                pred = int(s.argmax())
                if pred != lab:
                    W[lab] += lr * f
                    W[pred] -= lr * f
            else:
                p = F.softmax(s, dim=0)
                t = torch.zeros(10, device=device); t[lab] = 1.0
                err = t - p                                  # prediction error (surprise)
                w = 1.0 if rule == "surprise" else float(p.max())  # confidence scales C
                # delta-rule updates (err in [-1,1]) are smaller than the perceptron's fixed
                # kicks, so match the effective rate for a fair comparison
                W += DELTA_GAIN * lr * w * err.unsqueeze(1) * f.unsqueeze(0)
        trace.append((feat_te @ W.t()).argmax(1).eq(y_te).float().mean().item())
    return trace[-1], trace


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-exc", type=int, default=800)
    ap.add_argument("--feat-train", type=int, default=30000)
    ap.add_argument("--feat-passes", type=int, default=2)
    ap.add_argument("--readout-train", type=int, default=20000)
    ap.add_argument("--readout-epochs", type=int, default=15)
    ap.add_argument("--readout-lr", type=float, default=0.05)
    ap.add_argument("--n-test", type=int, default=10000)
    ap.add_argument("--T", type=int, default=100)
    ap.add_argument("--theta-plus", type=float, default=0.0005)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    print(f"device: {device} | n_exc: {args.n_exc} | surprise vs confidence learning")

    n_load = max(args.feat_train, args.readout_train)
    Xtr, ytr, Xte, yte = load_mnist(n_load, args.n_test)
    ytr_r, yte = ytr[:args.readout_train].to(device), yte.to(device)

    print("\n[1] train STDP features (frozen, shared by all arms)...")
    net = train_features(args.n_exc, Xtr[:args.feat_train], args.T, args.feat_passes,
                         args.theta_plus, device, args.seed)
    print("[2] extract frozen features...")
    feat_tr = extract(net, Xtr[:args.readout_train], args.T)
    feat_te = extract(net, Xte, args.T)

    print("[3] train each learning rule on the same features...")
    res = {}
    for rule in ("flat", "surprise", "confidence"):
        acc, trace = train_rule(rule, feat_tr, ytr_r, feat_te, yte,
                                args.readout_epochs, args.readout_lr, device)
        res[rule] = (acc, trace)
        # early-convergence marker: test acc after 1 and 3 epochs (sample efficiency)
        print(f"  {rule:11s} final {acc*100:.2f}%  |  ep1 {trace[0]*100:.1f}%  ep3 {trace[2]*100:.1f}%")

    print("\n" + "=" * 60)
    print(f"  confidence-into-learning  |  {args.n_exc} frozen features")
    print(f"  A flat (perceptron, current): {res['flat'][0]*100:.2f}%")
    print(f"  B surprise (delta / error):   {res['surprise'][0]*100:.2f}%")
    print(f"  C confidence (naive):         {res['confidence'][0]*100:.2f}%")
    print(f"  surprise - flat: {(res['surprise'][0]-res['flat'][0])*100:+.2f} pp")
    print(f"  confidence - flat: {(res['confidence'][0]-res['flat'][0])*100:+.2f} pp  "
          f"(expect <= 0: overflooding)")
    print("=" * 60)


if __name__ == "__main__":
    main()
