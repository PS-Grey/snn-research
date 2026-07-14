"""Surprise-weighted FEATURE learning — spend STDP where the net is wrong (three-factor, stable).

Exp 9 put surprise into the readout (89.18 -> 91.27%). This pushes it into the feature layer,
the higher-value but historically unstable direction (naive both-plastic collapsed -48pp via
anti-Hebbian dominance; "spike-rate as difficulty" is a recorded circular-instability failure).

Stability by design:
- Global surprise = a FROZEN critic readout's prediction error (1 - p_correct), one scalar per
  image. Not each neuron judging itself -> no circular feedback into the representation.
- MAGNITUDE only, never sign: feature STDP always moves toward the input (Hebbian, stable);
  surprise only scales how much. (The naive version flipped to anti-Hebbian on errors and
  scrambled features.)
- Self-limiting: as the net improves on hard images their surprise drops, so learning there
  slows -> negative feedback, not runaway.

Targets the Exp-7 plateau mechanism: pure STDP starves hard/rare classes; spending feature
learning on high-surprise images should reallocate neurons toward them.

Protocol: unsup features -> surprise-delta readout = BASELINE; surprise-weighted feature
fine-tune; fresh surprise-delta readout = FINAL. FINAL > BASELINE means it helped. No backprop.

Usage:
    .venv/bin/python experiments/stdp_surprise_features.py --n-exc 800 --feat-train 30000 \
        --ft-train 15000 --ft-epochs 2 --eta-ft 0.003
"""

import argparse
import time

import torch
import torch.nn.functional as F

from stdp_diehl_cook import STDPNet, load_mnist, W_MAX, NORM_TARGET

DELTA_GAIN = 10.0


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


def train_surprise_readout(feat, y, epochs, lr, device):
    """Delta-rule (surprise) readout — our best, Exp 9. Returns weights."""
    W = torch.zeros(10, feat.size(1), device=device)
    for _ in range(epochs):
        for idx in torch.randperm(feat.size(0)).tolist():
            f, lab = feat[idx], int(y[idx])
            p = F.softmax(W @ f, dim=0)
            t = torch.zeros(10, device=device); t[lab] = 1.0
            W += DELTA_GAIN * lr * (t - p).unsqueeze(1) * f.unsqueeze(0)
    return W


def eval_readout(net, X, y, W, T):
    return (extract(net, X, T) @ W.t()).argmax(1).eq(y).float().mean().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-exc", type=int, default=800)
    ap.add_argument("--feat-train", type=int, default=30000)
    ap.add_argument("--feat-passes", type=int, default=2)
    ap.add_argument("--ft-train", type=int, default=15000)
    ap.add_argument("--ft-epochs", type=int, default=2)
    ap.add_argument("--eta-ft", type=float, default=0.003)
    ap.add_argument("--readout-train", type=int, default=15000)
    ap.add_argument("--readout-epochs", type=int, default=20)
    ap.add_argument("--readout-lr", type=float, default=0.05)
    ap.add_argument("--n-test", type=int, default=10000)
    ap.add_argument("--T", type=int, default=100)
    ap.add_argument("--theta-plus", type=float, default=0.0005)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    print(f"device: {device} | n_exc: {args.n_exc} | surprise-weighted FEATURE learning")

    n_load = max(args.feat_train, args.ft_train, args.readout_train)
    Xtr, ytr, Xte, yte = load_mnist(n_load, args.n_test)
    yte = yte.to(device)

    print("\n[1] unsupervised STDP features...")
    net = train_unsup(args.n_exc, Xtr[:args.feat_train], args.T, args.feat_passes,
                      args.theta_plus, device, args.seed)

    print("[2] baseline: surprise-delta readout on unsupervised features...")
    feat0 = extract(net, Xtr[:args.readout_train], args.T)
    W0 = train_surprise_readout(feat0, ytr[:args.readout_train].to(device),
                                args.readout_epochs, args.readout_lr, device)
    base_acc = eval_readout(net, Xte, yte, W0, args.T)
    print(f"  BASELINE: {base_acc*100:.2f}%")

    print("[3] surprise-weighted feature fine-tuning (frozen critic W0, magnitude-only)...")
    yft = ytr[:args.ft_train].to(device)
    t0 = time.time()
    for ep in range(args.ft_epochs):
        surp_sum = 0.0
        for i in torch.randperm(args.ft_train).tolist():
            counts, elig = net.present_eligibility(Xtr[i], args.T)
            f = counts / counts.sum().clamp_min(1e-6)
            # binary surprise: 1 if the frozen critic gets it WRONG (hard example), else 0.
            # avoids the flat-softmax problem where 1-p_correct is ~constant regardless of
            # correctness; = pure hard-example mining (spend feature STDP only on errors)
            surprise = 0.0 if int((W0 @ f).argmax()) == int(yft[i]) else 1.0
            surp_sum += surprise
            net.W += args.eta_ft * surprise * elig          # Hebbian, magnitude by surprise
            net.W.clamp_(0.0, W_MAX)
            net.W *= (NORM_TARGET / net.W.sum(1, keepdim=True).clamp_min(1e-6))
        print(f"  ft epoch {ep+1}/{args.ft_epochs}  mean surprise {surp_sum/args.ft_train:.3f}"
              f"  ({time.time()-t0:.0f}s)")

    print("[4] fresh surprise-delta readout on fine-tuned features...")
    feat1 = extract(net, Xtr[:args.readout_train], args.T)
    W1 = train_surprise_readout(feat1, ytr[:args.readout_train].to(device),
                                args.readout_epochs, args.readout_lr, device)
    final_acc = eval_readout(net, Xte, yte, W1, args.T)

    print("\n" + "=" * 60)
    print(f"  surprise-weighted feature learning  |  {args.n_exc} neurons")
    print(f"  BASELINE (unsup features):          {base_acc*100:.2f}%")
    print(f"  FINAL (surprise-shaped features):   {final_acc*100:.2f}%")
    print(f"  delta: {(final_acc-base_acc)*100:+.2f} pp")
    print("=" * 60)


if __name__ == "__main__":
    main()
