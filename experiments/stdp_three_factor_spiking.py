"""Spike-faithful three-factor R-STDP readout — confirms the rate-form result (Exp 8).

Exp 8 got 89.18% with a rate-form R-STDP readout (feature spike-COUNTS -> reward-gated Hebbian).
This replaces the count collapse with a genuine time-resolved eligibility trace: the 10 output
neurons are LIF units driven by the frozen feature layer's spike TRAIN, each output synapse
accumulates an eligibility from real pre/post spike-timing coincidences over the trial, and a
global reward then gates the weight update from that eligibility. If this matches ~89%, the
rate form was a fair approximation; if not, time matters and the rate result was an artefact.

No backprop. Frozen unsupervised-STDP features; only the readout learns.

Usage:
    .venv/bin/python experiments/stdp_three_factor_spiking.py --n-exc 800 --feat-train 30000 \
        --readout-train 20000 --n-test 10000 --readout-epochs 20
"""

import argparse
import time

import torch

from stdp_diehl_cook import STDPNet, load_mnist

# readout LIF + eligibility dynamics
OUT_VDECAY = 0.9
OUT_THRESH = 0.15    # low enough that outputs fire several times per image (cold-start)
ELIG_DECAY = 0.95    # eligibility-trace decay (temporal credit window)
PRE_DECAY = 0.95     # pre-synaptic trace decay
OUT_INIT = 0.10      # weight init: outputs must fire from the start to get eligibility
TEACHER = 0.20       # training-only membrane boost to the target output (supervised R-STDP):
                     # forces the correct class to fire so its eligibility forms on the right
                     # features, solving the cold-start / WTA-monopoly credit failure


def train_features(n_exc, X, T, passes, theta_plus, device, seed):
    net = STDPNet(784, n_exc, device, seed=seed, theta_plus=theta_plus)
    t0 = time.time()
    for p in range(passes):
        for i in range(X.size(0)):
            net.present_image(X[i], T, learn=True)
            if (i + 1) % 5000 == 0:
                print(f"  [features] pass {p+1} img {i+1}/{X.size(0)} ({time.time()-t0:.0f}s)")
    return net


def extract_trains(net, X, T):
    """Cache each image's frozen feature spike train (winner idx per timestep) as a LongTensor."""
    trains = torch.full((X.size(0), T), -1, dtype=torch.long)
    for i in range(X.size(0)):
        trains[i] = net.present_spiketrain(X[i], T).cpu()
    return trains


def run_readout(W_out, train, T, n_exc, device, learn=False, label=None, lr=0.0):
    """Run the spiking readout over one feature spike train.
    Output LIF neurons integrate feature spikes; eligibility accumulates pre-trace on each output
    spike. Returns predicted class. If learn, applies reward-gated update from the eligibility."""
    n_cls = W_out.size(0)
    v = torch.zeros(n_cls, device=device)
    x_pre = torch.zeros(n_exc, device=device)          # feature (pre) trace
    elig = torch.zeros(n_cls, n_exc, device=device)    # per-synapse eligibility
    counts = torch.zeros(n_cls, device=device)
    for t in range(T):
        w = int(train[t])
        if w >= 0:
            x_pre.mul_(PRE_DECAY)
            x_pre[w] += 1.0
            v = v * OUT_VDECAY + W_out[:, w]           # only the active feature drives outputs
        else:
            v = v * OUT_VDECAY
        if learn:
            v[label] += TEACHER                        # teacher forcing: drive the target output
        elig.mul_(ELIG_DECAY)
        above = v - OUT_THRESH
        if (above > 0).any():
            # output-layer WTA: single winner fires (competition, as in Mozafari R-STDP), so
            # eligibility concentrates on the responsible output rather than smearing over all 10
            k = int(above.argmax())
            counts[k] += 1
            v[k] = 0.0
            elig[k] += x_pre                            # pre×post coincidence -> eligibility
    pred = int(counts.argmax()) if counts.sum() > 0 else int(v.argmax())
    if learn:
        # three-factor: global reward gates the eligibility-driven update
        W_out[label] += lr * elig[label]               # reward the target class
        if pred != label:
            W_out[pred] -= lr * elig[pred]             # punish the wrong winner
        W_out.clamp_(min=0.0)
    return pred, counts.sum().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-exc", type=int, default=800)
    ap.add_argument("--feat-train", type=int, default=30000)
    ap.add_argument("--feat-passes", type=int, default=2)
    ap.add_argument("--readout-train", type=int, default=20000)
    ap.add_argument("--n-test", type=int, default=10000)
    ap.add_argument("--readout-epochs", type=int, default=20)
    ap.add_argument("--readout-lr", type=float, default=0.02)
    ap.add_argument("--T", type=int, default=100)
    ap.add_argument("--theta-plus", type=float, default=0.0005)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    print(f"device: {device} | n_exc: {args.n_exc} | spike-faithful readout")

    n_load = max(args.feat_train, args.readout_train)
    Xtr, ytr, Xte, yte = load_mnist(n_load, args.n_test)

    print("\n[1/3] training unsupervised STDP feature layer (frozen after)...")
    net = train_features(args.n_exc, Xtr[:args.feat_train], args.T, args.feat_passes,
                         args.theta_plus, device, args.seed)

    print("\n[2/3] caching frozen feature spike trains...")
    t0 = time.time()
    tr_trains = extract_trains(net, Xtr[:args.readout_train], args.T)
    te_trains = extract_trains(net, Xte, args.T)
    ytr_r, yte_l = ytr[:args.readout_train].tolist(), yte.tolist()
    print(f"  cached in {time.time()-t0:.0f}s")

    print("\n[3/3] training spike-faithful eligibility-trace R-STDP readout...")
    # small positive init so output neurons fire from the start (cold-start fix)
    W_out = torch.rand(10, args.n_exc, device=device) * OUT_INIT
    N = tr_trains.size(0)
    for ep in range(args.readout_epochs):
        perm = torch.randperm(N)
        errs, spk = 0, 0.0
        for j in perm.tolist():
            tr = tr_trains[j].to(device)
            pred, nspk = run_readout(W_out, tr, args.T, args.n_exc, device,
                                     learn=True, label=ytr_r[j], lr=args.readout_lr)
            errs += (pred != ytr_r[j])
            spk += nspk
        print(f"  [readout] epoch {ep+1}/{args.readout_epochs}  train err {errs/N*100:.2f}%"
              f"  out-spikes/img {spk/N:.1f}")

    correct = 0
    for j in range(te_trains.size(0)):
        pred, _ = run_readout(W_out, te_trains[j].to(device), args.T, args.n_exc, device, learn=False)
        correct += (pred == yte_l[j])
    acc = correct / te_trains.size(0)

    print("\n" + "=" * 60)
    print(f"  spike-faithful three-factor R-STDP  |  {args.n_exc} frozen features")
    print(f"  test accuracy: {acc*100:.2f}%")
    print(f"  (rate-form Exp 8: 89.18% | pure-STDP 800: 86.04% | surrogate ref: 98.9%)")
    print("=" * 60)


if __name__ == "__main__":
    main()
