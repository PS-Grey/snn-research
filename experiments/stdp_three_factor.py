"""Three-factor R-STDP readout on frozen unsupervised-STDP features.

The pure-STDP scaling curve plateaus at ~88.6% on MNIST and stops responding to more neurons,
because unsupervised STDP has no signal to allocate capacity by class usefulness. This adds the
missing third factor: a global reward signal that gates a local plasticity update, the
hardware-native (Loihi-supported) form of goal-directed learning. Still no backprop.

Design (features frozen, only the readout learns):
1. Train the unsupervised STDP feature layer (stdp_diehl_cook.STDPNet), then FREEZE it.
2. Extract a feature vector per image = per-neuron spike counts over one frozen presentation.
3. Train a 10-class spiking readout by **reward-modulated STDP** (three-factor, rate form):
   each synapse has a local pre×post eligibility; a global reward (correct vs wrong) gates
   whether it potentiates or depresses. On error: push the correct class up, the wrong winner
   down (reward-gated Hebbian) — the on-chip R-STDP classification rule (Mozafari-style).

Measured against the pure-STDP floor (~88.6%) and the surrogate-gradient ceiling (98.9%).

Usage:
    .venv/bin/python experiments/stdp_three_factor.py --n-exc 800 --feat-train 30000 \
        --readout-train 20000 --n-test 10000 --readout-epochs 15
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
            if (i + 1) % 5000 == 0:
                print(f"  [features] pass {p+1} img {i+1}/{X.size(0)} ({time.time()-t0:.0f}s)")
    return net


def extract_features(net, X, T):
    """Per-image feature vector = frozen-net spike counts per neuron, L1-normalised."""
    feats = torch.zeros(X.size(0), net.n_exc, device=net.device)
    for i in range(X.size(0)):
        feats[i] = net.present_image(X[i], T, learn=False)
    feats = feats / feats.sum(1, keepdim=True).clamp_min(1e-6)  # normalise firing budget per image
    return feats


def train_rstdp_readout(feat, y, n_classes, epochs, lr, device):
    """Reward-modulated (three-factor) readout. W_out: [n_classes, n_feat].
    Perceptron-style R-STDP: predict by argmax; on error, reward-gate a local Hebbian update
    (potentiate the correct class's synapses on this input, depress the wrong winner's)."""
    W = torch.zeros(n_classes, feat.size(1), device=device)
    N = feat.size(0)
    for ep in range(epochs):
        perm = torch.randperm(N, device=device)
        errs = 0
        for idx in perm:
            f = feat[idx]
            pred = int((W @ f).argmax())
            lab = int(y[idx])
            if pred != lab:                      # reward = -1 (wrong): correct the update
                W[lab] += lr * f                 # eligibility (pre×post) gated by reward sign
                W[pred] -= lr * f
                errs += 1
        print(f"  [readout] epoch {ep+1}/{epochs}  train err {errs/N*100:.2f}%")
    return W


def evaluate_readout(W, feat, y):
    pred = (feat @ W.t()).argmax(1)
    return (pred == y).float().mean().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-exc", type=int, default=800)
    ap.add_argument("--feat-train", type=int, default=30000)   # imgs to train the feature layer
    ap.add_argument("--feat-passes", type=int, default=2)
    ap.add_argument("--readout-train", type=int, default=20000)  # imgs for the readout
    ap.add_argument("--n-test", type=int, default=10000)
    ap.add_argument("--readout-epochs", type=int, default=15)
    ap.add_argument("--readout-lr", type=float, default=0.01)
    ap.add_argument("--T", type=int, default=100)
    ap.add_argument("--theta-plus", type=float, default=0.0005)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    print(f"device: {device} | n_exc: {args.n_exc} | feat_train: {args.feat_train} "
          f"| readout_train: {args.readout_train}")

    n_load = max(args.feat_train, args.readout_train)
    Xtr, ytr, Xte, yte = load_mnist(n_load, args.n_test)

    print("\n[1/3] training unsupervised STDP feature layer (frozen after)...")
    net = train_features(args.n_exc, Xtr[:args.feat_train], args.T, args.feat_passes,
                         args.theta_plus, device, args.seed)

    print("\n[2/3] extracting frozen features...")
    t0 = time.time()
    feat_tr = extract_features(net, Xtr[:args.readout_train], args.T)
    feat_te = extract_features(net, Xte, args.T)
    ytr_r, yte = ytr[:args.readout_train].to(device), yte.to(device)
    print(f"  extracted in {time.time()-t0:.0f}s")

    print("\n[3/3] training three-factor R-STDP readout...")
    W = train_rstdp_readout(feat_tr, ytr_r, 10, args.readout_epochs, args.readout_lr, device)
    acc = evaluate_readout(W, feat_te, yte)

    print("\n" + "=" * 60)
    print(f"  three-factor R-STDP  |  {args.n_exc} frozen STDP features")
    print(f"  test accuracy: {acc*100:.2f}%")
    print(f"  (pure-STDP floor {args.n_exc}n: ~86-88.6% | surrogate ref: 98.9%)")
    print("=" * 60)


if __name__ == "__main__":
    main()
