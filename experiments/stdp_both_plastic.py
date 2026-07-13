"""Both-layers-plastic three-factor — reward shapes the FEATURES, not just the readout.

Exp 8 rewarded only the readout on frozen unsupervised features (89.18%), capped by feature
separability. This lets the global reward reshape the feature layer too: reward-modulated STDP.
Each winning feature neuron accumulates an STDP eligibility over the trial; a global reward
(correct/wrong classification) then gates the feature update — correct decisions reinforce the
responsible features, errors push them away from the input that fooled them. Still no backprop.

Protocol (clean measurement — does reward-shaping the features improve their separability?):
1. Unsupervised STDP features -> fresh rate-form readout -> BASELINE acc (~89%).
2. Reward-modulated fine-tune of the features (online readout supplies the reward signal).
3. Freeze fine-tuned features -> re-extract -> fresh readout -> FINAL acc.
If FINAL > BASELINE, the third factor reshaping features closed some of the gap.

Usage:
    .venv/bin/python experiments/stdp_both_plastic.py --n-exc 800 --feat-train 30000 \
        --ft-train 15000 --ft-epochs 2 --eta-ft 0.002
"""

import argparse
import time

import torch

from stdp_diehl_cook import STDPNet, load_mnist, W_MAX, NORM_TARGET


def train_unsup(n_exc, X, T, passes, theta_plus, device, seed):
    net = STDPNet(784, n_exc, device, seed=seed, theta_plus=theta_plus)
    t0 = time.time()
    for p in range(passes):
        for i in range(X.size(0)):
            net.present_image(X[i], T, learn=True)
            if (i + 1) % 10000 == 0:
                print(f"  [unsup] pass {p+1} img {i+1}/{X.size(0)} ({time.time()-t0:.0f}s)")
    return net


def extract(net, X, T):
    feats = torch.zeros(X.size(0), net.n_exc, device=net.device)
    for i in range(X.size(0)):
        feats[i] = net.present_image(X[i], T, learn=False)
    return feats / feats.sum(1, keepdim=True).clamp_min(1e-6)


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


def eval_readout(net, X, y, W, T):
    feat = extract(net, X, T)
    return (feat @ W.t()).argmax(1).eq(y).float().mean().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-exc", type=int, default=800)
    ap.add_argument("--feat-train", type=int, default=30000)
    ap.add_argument("--feat-passes", type=int, default=2)
    ap.add_argument("--ft-train", type=int, default=15000)     # images for reward fine-tuning
    ap.add_argument("--ft-epochs", type=int, default=2)
    ap.add_argument("--eta-ft", type=float, default=0.002)     # feature reward-STDP rate (gentle)
    ap.add_argument("--readout-train", type=int, default=15000)
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
    print(f"device: {device} | n_exc: {args.n_exc} | both-layers-plastic")

    n_load = max(args.feat_train, args.ft_train, args.readout_train)
    Xtr, ytr, Xte, yte = load_mnist(n_load, args.n_test)
    yte = yte.to(device)

    print("\n[1] unsupervised STDP features...")
    net = train_unsup(args.n_exc, Xtr[:args.feat_train], args.T, args.feat_passes,
                      args.theta_plus, device, args.seed)

    print("[2] baseline readout on unsupervised features...")
    feat0 = extract(net, Xtr[:args.readout_train], args.T)
    W0 = train_readout(feat0, ytr[:args.readout_train].to(device),
                       args.readout_epochs, args.readout_lr, device)
    base_acc = eval_readout(net, Xte, yte, W0, args.T)
    print(f"  BASELINE test acc: {base_acc*100:.2f}%")

    print("[3] reward-modulated feature fine-tuning (three-factor, both plastic)...")
    # W0 is a FROZEN critic: a stable ~baseline-accuracy reward source, so ~80% of images give
    # +1 (reinforce) and only ~20% give -1 (anti-Hebbian) — avoids the negative-reward collapse.
    yft = ytr[:args.ft_train].to(device)
    t0 = time.time()
    for ep in range(args.ft_epochs):
        errs = 0
        for i in torch.randperm(args.ft_train).tolist():
            counts, elig = net.present_eligibility(Xtr[i], args.T)
            f = counts / counts.sum().clamp_min(1e-6)
            lab = int(yft[i])
            pred = int((W0 @ f).argmax())            # frozen critic supplies the reward
            R = 1.0 if pred == lab else -1.0
            errs += (pred != lab)
            # feature update (reward-gated STDP): correct -> reinforce, wrong -> push away
            net.W += args.eta_ft * R * elig
            net.W.clamp_(0.0, W_MAX)
            net.W *= (NORM_TARGET / net.W.sum(1, keepdim=True).clamp_min(1e-6))
        print(f"  ft epoch {ep+1}/{args.ft_epochs}  critic err {errs/args.ft_train*100:.2f}%"
              f"  ({time.time()-t0:.0f}s)")

    print("[4] fresh readout on fine-tuned features...")
    feat1 = extract(net, Xtr[:args.readout_train], args.T)
    W1 = train_readout(feat1, ytr[:args.readout_train].to(device),
                       args.readout_epochs, args.readout_lr, device)
    final_acc = eval_readout(net, Xte, yte, W1, args.T)

    print("\n" + "=" * 60)
    print(f"  both-layers-plastic three-factor  |  {args.n_exc} neurons")
    print(f"  BASELINE (unsup features + readout): {base_acc*100:.2f}%")
    print(f"  FINAL (reward-shaped features + readout): {final_acc*100:.2f}%")
    print(f"  delta: {(final_acc-base_acc)*100:+.2f} pp")
    print("=" * 60)


if __name__ == "__main__":
    main()
