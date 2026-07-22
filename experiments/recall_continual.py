"""Does recall fix the forgetting? Exemplar-rehearsal A/B (Exp 17).

Exp 16 diagnosed catastrophic forgetting as READOUT suppression: new-task training pushes old-class
outputs to zero. Recall/rehearsal should attack this directly — keep old classes present so their
outputs aren't suppressed. This is the confirmation step (a small stored exemplar buffer); the
novel version (EP-native GENERATIVE recall + forgetting-curve scheduling) is the follow-up.

Class-incremental MNIST (5 tasks x 2 classes), EP. Two arms:
  no-recall  — Exp 16 baseline (task-0 collapses to ~6%)
  recall     — keep k exemplars per seen class, mix them into each new-task batch

If recall restores task-0 retention, the mechanism (and the direction) is confirmed.

Usage: .venv/bin/python experiments/recall_continual.py --epochs 8 --buffer-k 20
"""

import argparse
import time

import torch
import torch.nn.functional as F
from torchvision import datasets, transforms

from equilibrium_prop import EPNet, pick_device

TASKS = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]


def run(recall, buf_k, Xtr, ytr, Xte, yte, hidden, epochs, batch, beta, steps, dev):
    net = EPNet(hidden, dev)
    bufX = torch.empty(0, 784, device=dev); bufY = torch.empty(0, dtype=torch.long, device=dev)
    curve = []
    for ti, cls in enumerate(TASKS):
        m = torch.zeros_like(ytr, dtype=torch.bool)
        for c in cls:
            m |= (ytr == c)
        Xt, yt = Xtr[m], ytr[m]
        for _ in range(epochs):
            perm = torch.randperm(len(Xt), device=dev)
            for b in range(0, len(Xt), batch):
                idx = perm[b:b+batch]
                x, y = Xt[idx], yt[idx]
                if recall and len(bufX):                       # mix in rehearsed old classes
                    r = torch.randint(0, len(bufX), (min(batch, len(bufX)),), device=dev)
                    x = torch.cat([x, bufX[r]]); y = torch.cat([y, bufY[r]])
                net.train_step(x, F.one_hot(y, 10).float(), beta, steps)
        # store k exemplars per class just learned
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
    ap.add_argument("--hidden", type=int, default=500)
    ap.add_argument("--steps", type=int, default=25)
    ap.add_argument("--beta", type=float, default=0.5)
    ap.add_argument("--buffer-k", type=int, default=20)
    ap.add_argument("--n-train", type=int, default=60000)
    ap.add_argument("--n-test", type=int, default=10000)
    args = ap.parse_args()

    dev = pick_device(); torch.manual_seed(0)
    print(f"device: {dev} | recall A/B (EP, class-incremental) | buffer {args.buffer_k}/class")
    tf = transforms.ToTensor()
    tr = datasets.MNIST("./data", train=True, download=True, transform=tf)
    te = datasets.MNIST("./data", train=False, download=True, transform=tf)
    Xtr = tr.data[:args.n_train].float().view(-1, 784).to(dev)/255.0; ytr = tr.targets[:args.n_train].to(dev)
    Xte = te.data[:args.n_test].float().view(-1, 784).to(dev)/255.0; yte = te.targets[:args.n_test].to(dev)

    for recall in (False, True):
        t0 = time.time()
        curve, final = run(recall, args.buffer_k, Xtr, ytr, Xte, yte, args.hidden,
                           args.epochs, args.batch, args.beta, args.steps, dev)
        tag = "recall   " if recall else "no-recall"
        print(f"\n[{tag}]  task-0 acc after each stage:  " + "  ".join(f"{a*100:4.0f}" for a in curve))
        print(f"        task-0: {curve[0]*100:.0f}% -> {curve[-1]*100:.0f}%   "
              f"final all-class {final*100:.1f}%   ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
