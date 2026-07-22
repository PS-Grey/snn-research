"""Self-measured forgetting-scheduled replay — the genuinely-open piece (Exp 18).

Lit-scout (2026-07-22) verdict: EP settle-to-generate rehearsal is now published (Cook et al.
2026), but scheduling replay by the model's OWN measured forgetting is OPEN. This tests that open
claim directly, at MATCHED replay budget (the comparison no paper reports):

  uniform    — replay old classes uniformly (the standard baseline)
  forgetting — measure each old class's current retention on the buffer, replay classes in
               proportion to their forgetting RISK (about-to-fade classes get more of the budget)

Both arms get the SAME small replay budget per step, so the only difference is WHICH classes the
budget is spent on. If forgetting-scheduled > uniform at matched budget, the self-measured
schedule earns its keep. (Energy/basin-depth as the forgetting proxy, and spiking, are the next
steps; here the forgetting signal is measured retention, the cleanest first version.)

Class-incremental MNIST (5 tasks x 2), EP. Usage:
    .venv/bin/python experiments/recall_scheduled.py --epochs 8 --budget 16 --buffer-k 20
"""

import argparse
import time

import torch
import torch.nn.functional as F
from torchvision import datasets, transforms

from equilibrium_prop import EPNet, pick_device

TASKS = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]


def class_retention(net, bufX, bufY, classes, steps):
    """The self-measured forgetting signal: current accuracy per class on its buffer exemplars."""
    acc = {}
    for c in classes:
        m = bufY == c
        if m.any():
            acc[c] = (net.predict(bufX[m], steps) == c).float().mean().item()
    return acc


def run(schedule, budget, buf_k, Xtr, ytr, Xte, yte, hidden, epochs, batch, beta, steps, dev):
    net = EPNet(hidden, dev)
    bufX = torch.empty(0, 784, device=dev); bufY = torch.empty(0, dtype=torch.long, device=dev)
    curve = []
    for ti, cls in enumerate(TASKS):
        m = torch.zeros_like(ytr, dtype=torch.bool)
        for c in cls:
            m |= (ytr == c)
        Xt, yt = Xtr[m], ytr[m]
        seen = torch.unique(bufY).tolist()
        for ep in range(epochs):
            # self-measured forgetting weights, refreshed each epoch (uniform arm ignores them)
            if schedule == "forgetting" and seen:
                ret = class_retention(net, bufX, bufY, seen, steps)
                w = torch.tensor([ (1.0 - ret.get(int(bufY[i]), 1.0)) + 0.05 for i in range(len(bufY)) ],
                                 device=dev)
            else:
                w = torch.ones(len(bufY), device=dev)
            perm = torch.randperm(len(Xt), device=dev)
            for b in range(0, len(Xt), batch):
                idx = perm[b:b+batch]
                x, y = Xt[idx], yt[idx]
                if len(bufX):                                      # matched replay budget
                    r = torch.multinomial(w, min(budget, len(bufX)), replacement=len(bufX) < budget)
                    x = torch.cat([x, bufX[r]]); y = torch.cat([y, bufY[r]])
                net.train_step(x, F.one_hot(y, 10).float(), beta, steps)
        for c in cls:
            ci = (yt == c).nonzero(as_tuple=True)[0][:buf_k]
            bufX = torch.cat([bufX, Xt[ci]]); bufY = torch.cat([bufY, yt[ci]])
        m0 = (yte == 0) | (yte == 1)
        curve.append((net.predict(Xte[m0], steps) == yte[m0]).float().mean().item())
    final = sum((net.predict(Xte[b:b+500], steps) == yte[b:b+500]).sum().item()
                for b in range(0, len(Xte), 500)) / len(Xte)
    return curve, final


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--budget", type=int, default=16)   # replay samples per step (matched)
    ap.add_argument("--buffer-k", type=int, default=20)
    ap.add_argument("--hidden", type=int, default=500)
    ap.add_argument("--steps", type=int, default=25)
    ap.add_argument("--beta", type=float, default=0.5)
    ap.add_argument("--n-train", type=int, default=60000)
    ap.add_argument("--n-test", type=int, default=10000)
    args = ap.parse_args()

    dev = pick_device(); torch.manual_seed(0)
    print(f"device: {dev} | scheduled replay | budget {args.budget}/step, buffer {args.buffer_k}/class")
    tf = transforms.ToTensor()
    tr = datasets.MNIST("./data", train=True, download=True, transform=tf)
    te = datasets.MNIST("./data", train=False, download=True, transform=tf)
    Xtr = tr.data[:args.n_train].float().view(-1, 784).to(dev)/255.0; ytr = tr.targets[:args.n_train].to(dev)
    Xte = te.data[:args.n_test].float().view(-1, 784).to(dev)/255.0; yte = te.targets[:args.n_test].to(dev)

    for schedule in ("uniform", "forgetting"):
        t0 = time.time()
        curve, final = run(schedule, args.budget, args.buffer_k, Xtr, ytr, Xte, yte, args.hidden,
                           args.epochs, args.batch, args.beta, args.steps, dev)
        print(f"\n[{schedule:10s}]  task-0 stages: " + "  ".join(f"{a*100:4.0f}" for a in curve))
        print(f"             task-0 {curve[0]*100:.0f}%->{curve[-1]*100:.0f}%   "
              f"final all-class {final*100:.1f}%   ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
