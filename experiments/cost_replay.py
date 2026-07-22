"""Cost-of-relearning replay scheduling — scout-confirmed OPEN (Exp 21).

Lit-scout (2026-07-22) verdict: scheduling replay by REACQUISITION COST (how expensive a memory is
to relearn if dropped) rather than by FORGETTING MAGNITUDE (how degraded it is now) is an open,
unbuilt replay-selection criterion -- distinct from MIR (interference), gradient-coreset
(importance), diversity; grounded in Ebbinghaus savings + Bjork storage/retrieval strength; NO
spiking/neuromorphic/local implementation exists.

This is the first such implementation. Three arms at MATCHED replay budget on 20-class MNIST+Fashion
class-incremental (EP):
    uniform  -- replay old classes uniformly
    mir      -- replay in proportion to FORGETTING: low current buffer retention -> more replay
                (the interference baseline the scout names to beat)
    cost     -- replay in proportion to REACQUISITION COST: narrow-basin/fragile classes -> more
                replay. Cost proxy = basin NARROWNESS: perturb a buffer exemplar's input, re-settle,
                measure how much its class margin collapses. Fragile = expensive to reacquire. This
                is the scout's on-chip proxy (basin width / settling), EP-native, no ceiling confound.

Scout prediction: cost beats mir on HARD / low-redundancy classes (here: Fashion) and on WORST-class
accuracy, because mir under-weights classes that are costly-but-not-currently-forgetting. So the
decisive metrics are Fashion-mean and worst-class acc, NOT overall acc.

Usage: .venv/bin/python experiments/cost_replay.py --epochs 6 --budget 16 --buffer-k 20
"""

import argparse
import time

import torch
import torch.nn.functional as F

from equilibrium_prop import EPNet, pick_device, rho
from continual_hetero import NC, TASKS, load_2020, per_class_acc


@torch.no_grad()
def class_margin(net, X, steps):
    """Correct-vs-runner-up output margin per sample (needs labels via caller)."""
    _, y = net.settle(X, None, 0.0, steps)
    return rho(y)


@torch.no_grad()
def basin_narrowness(net, bufX, bufY, classes, steps, noise=0.15):
    """Reacquisition-cost proxy: margin collapse under a small input perturbation. A fragile,
    narrow basin (large collapse) is expensive to reacquire; a wide/robust one is cheap."""
    cost = {}
    for c in classes:
        m = bufY == c
        if not m.any():
            continue
        Xc = bufX[m]
        y0 = rho(net.settle(Xc, None, 0.0, steps)[1])
        y1 = rho(net.settle((Xc + noise * torch.randn_like(Xc)).clamp(0, 1), None, 0.0, steps)[1])
        others = torch.ones(NC, device=Xc.device, dtype=torch.bool); others[c] = False
        marg0 = y0[:, c] - y0[:, others].max(1).values
        marg1 = y1[:, c] - y1[:, others].max(1).values
        cost[c] = (marg0 - marg1).clamp(min=0).mean().item()   # margin collapse = fragility
    return cost


def replay_weights(schedule, net, bufX, bufY, seen, steps, dev):
    if not seen or schedule in ("none", "uniform"):
        return torch.ones(len(bufY), device=dev)
    if schedule == "mir":                                       # forgetting: low retention -> replay
        r = per_class_acc(net, bufX, bufY, seen, steps)
        return torch.tensor([(1.0 - r.get(int(bufY[i]), 1.0)) + 0.05 for i in range(len(bufY))],
                            device=dev)
    s = basin_narrowness(net, bufX, bufY, seen, steps)          # cost: fragile -> replay
    lo, hi = min(s.values()), max(s.values()); rng = (hi - lo) or 1.0
    norm = {c: (s[c] - lo) / rng for c in s}
    return torch.tensor([norm.get(int(bufY[i]), 0.5) + 0.05 for i in range(len(bufY))], device=dev)


def run(schedule, budget, buf_k, Xtr, ytr, Xte, yte, hidden, epochs, batch, beta, steps, dev, seed=0):
    net = EPNet(hidden, dev, n_classes=NC, seed=seed)
    bufX = torch.empty(0, 784, device=dev); bufY = torch.empty(0, dtype=torch.long, device=dev)
    for cls in TASKS:
        m = torch.zeros_like(ytr, dtype=torch.bool)
        for c in cls:
            m |= (ytr == c)
        Xt, yt = Xtr[m], ytr[m]
        seen = torch.unique(bufY).tolist()
        for _ in range(epochs):
            w = replay_weights(schedule, net, bufX, bufY, seen, steps, dev)
            perm = torch.randperm(len(Xt), device=dev)
            for b in range(0, len(Xt), batch):
                idx = perm[b:b+batch]
                x, y = Xt[idx], yt[idx]
                if len(bufX):
                    r = torch.multinomial(w, min(budget, len(bufX)), replacement=len(bufX) < budget)
                    x = torch.cat([x, bufX[r]]); y = torch.cat([y, bufY[r]])
                net.train_step(x, F.one_hot(y, NC).float(), beta, steps)
        for c in cls:
            ci = (yt == c).nonzero(as_tuple=True)[0][:buf_k]
            bufX = torch.cat([bufX, Xt[ci]]); bufY = torch.cat([bufY, yt[ci]])
    return per_class_acc(net, Xte, yte, list(range(NC)), steps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--budget", type=int, default=16)
    ap.add_argument("--buffer-k", type=int, default=20)
    ap.add_argument("--hidden", type=int, default=500)
    ap.add_argument("--steps", type=int, default=25)
    ap.add_argument("--beta", type=float, default=0.5)
    ap.add_argument("--per-class", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    dev = pick_device(); torch.manual_seed(args.seed)
    print(f"device: {dev} | cost-of-relearning replay | 20-class MNIST+Fashion | "
          f"budget {args.budget}/step, buffer {args.buffer_k}/class")
    Xtr, ytr, Xte, yte = load_2020(args.per_class, dev)

    import statistics as st
    print(f"\n{'arm':10s} {'overall':>8s} {'MNIST':>7s} {'Fashion':>8s} {'worst':>7s}")
    for schedule in ("uniform", "mir", "cost"):
        t0 = time.time()
        final = run(schedule, args.budget, args.buffer_k, Xtr, ytr, Xte, yte, args.hidden,
                    args.epochs, args.batch, args.beta, args.steps, dev, args.seed)
        overall = st.mean(final.values())
        mnist = st.mean(final[c] for c in range(10))
        fashion = st.mean(final[c] for c in range(10, 20))
        worst = min(final.values())
        print(f"{schedule:10s} {overall*100:7.1f}% {mnist*100:6.1f}% {fashion*100:7.1f}% "
              f"{worst*100:6.1f}%  ({time.time()-t0:.0f}s)")
    print("\nscout prediction: cost >= mir on Fashion (hard/expensive) and worst-class; may trade "
          "a little MNIST (easy/cheap-to-relearn). Decisive metrics = Fashion + worst, not overall.")


if __name__ == "__main__":
    main()
