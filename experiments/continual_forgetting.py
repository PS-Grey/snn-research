"""Catastrophic forgetting demo — does on-chip continual learning keep old knowledge? (Exp 15)

Answers Sergiy's question directly: if we train on MNIST then show Fashion-MNIST, does it handle
both, or forget MNIST? Uses our working EP net (Exp 13). Train on MNIST, test MNIST; then train
on Fashion, RE-test MNIST (expected to drop = forgetting) + test Fashion.

This sets up the novel direction: on-chip continual learning WITHOUT forgetting -- where a learned
graded-payload channel and/or stickiness (metaplasticity) could genuinely help, and where
neuromorphic could beat a frozen GPU-trained chip.

Usage: .venv/bin/python experiments/continual_forgetting.py --epochs 10
"""

import argparse
import time

import torch
import torch.nn.functional as F
from torchvision import datasets, transforms

from equilibrium_prop import EPNet, pick_device


def load(kind, n_tr, n_te, dev):
    tf = transforms.ToTensor()
    D = datasets.FashionMNIST if kind == "fashion" else datasets.MNIST
    tr = D("./data", train=True, download=True, transform=tf)
    te = D("./data", train=False, download=True, transform=tf)
    Xtr = tr.data[:n_tr].float().view(-1, 784).to(dev) / 255.0
    ytr = tr.targets[:n_tr].to(dev)
    Xte = te.data[:n_te].float().view(-1, 784).to(dev) / 255.0
    yte = te.targets[:n_te].to(dev)
    return Xtr, ytr, Xte, yte


def train(net, Xtr, ytr, epochs, batch, beta, steps):
    for _ in range(epochs):
        perm = torch.randperm(len(Xtr), device=Xtr.device)
        for b in range(0, len(Xtr), batch):
            idx = perm[b:b+batch]
            net.train_step(Xtr[idx], F.one_hot(ytr[idx], 10).float(), beta, steps)


def test(net, Xte, yte, steps):
    return sum((net.predict(Xte[b:b+500], steps) == yte[b:b+500]).sum().item()
               for b in range(0, len(Xte), 500)) / len(Xte)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--hidden", type=int, default=500)
    ap.add_argument("--steps", type=int, default=25)
    ap.add_argument("--beta", type=float, default=0.5)
    ap.add_argument("--n-train", type=int, default=60000)
    ap.add_argument("--n-test", type=int, default=10000)
    args = ap.parse_args()

    dev = pick_device(); torch.manual_seed(0)
    print(f"device: {dev} | catastrophic-forgetting demo (EP, hidden {args.hidden})")
    Mtr, Mytr, Mte, Myte = load("mnist", args.n_train, args.n_test, dev)
    Ftr, Fytr, Fte, Fyte = load("fashion", args.n_train, args.n_test, dev)

    net = EPNet(args.hidden, dev)

    print("\n[phase 1] train on MNIST...")
    t0 = time.time(); train(net, Mtr, Mytr, args.epochs, args.batch, args.beta, args.steps)
    m1 = test(net, Mte, Myte, args.steps)
    print(f"  MNIST acc after phase 1: {m1*100:.2f}%  ({time.time()-t0:.0f}s)")

    print("[phase 2] now train on Fashion-MNIST (no MNIST)...")
    t0 = time.time(); train(net, Ftr, Fytr, args.epochs, args.batch, args.beta, args.steps)
    m2 = test(net, Mte, Myte, args.steps)
    f2 = test(net, Fte, Fyte, args.steps)
    print(f"  Fashion acc after phase 2: {f2*100:.2f}%  ({time.time()-t0:.0f}s)")
    print(f"  MNIST acc after phase 2:   {m2*100:.2f}%   (was {m1*100:.2f}%)")

    print("\n" + "=" * 56)
    print(f"  MNIST: {m1*100:.1f}% -> {m2*100:.1f}%   ({(m2-m1)*100:+.1f} pp = FORGETTING)")
    print(f"  learned Fashion ({f2*100:.1f}%) by overwriting MNIST." if m2 < m1 - 5
          else "  retained both (unexpected).")
    print("=" * 56)


if __name__ == "__main__":
    main()
