"""Does the learning rule affect forgetting? Class-incremental comparison (Exp 16).

Sergiy's question: are STDP / EP / etc. hit by catastrophic forgetting the same, or differently?
Hypothesis: forgetting happens WHERE the plasticity is. End-to-end plastic (EP) overwrites the
shared representation and forgets hard; a FROZEN task-agnostic representation (only the readout
adapts) should retain old classes far better.

Class-incremental testbed (Sergiy's idea — cleaner than MNIST-then-Fashion): MNIST 10 classes as
5 tasks of 2 classes each. Train task by task; after each, test on ALL classes seen so far.
Forgetting = how much the first task's accuracy decays as later tasks arrive.

Two arms (same EP net, isolating the mechanism):
  full   — everything plastic (proxy for end-to-end methods like EP)
  frozen — feature layer frozen after task 0, only the readout adapts (proxy for frozen
           unsupervised features + readout, i.e. the STDP-style methods)

Usage: .venv/bin/python experiments/continual_compare.py --epochs 8
"""

import argparse
import time

import torch
import torch.nn.functional as F
from torchvision import datasets, transforms

from equilibrium_prop import EPNet, pick_device

TASKS = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]


def run(arm, Xtr, ytr, Xte, yte, hidden, epochs, batch, beta, steps, dev):
    net = EPNet(hidden, dev)
    seen = []
    acc_task0 = []   # accuracy on task-0 classes ({0,1}) after each stage
    for ti, cls in enumerate(TASKS):
        seen = seen + cls
        m = torch.zeros_like(ytr, dtype=torch.bool)
        for c in cls:
            m |= (ytr == c)
        Xt, yt = Xtr[m], ytr[m]
        freeze = (arm == "frozen" and ti > 0)
        for _ in range(epochs):
            perm = torch.randperm(len(Xt), device=dev)
            for b in range(0, len(Xt), batch):
                idx = perm[b:b+batch]
                net.train_step(Xt[idx], F.one_hot(yt[idx], 10).float(), beta, steps, freeze)
        # accuracy on task-0 classes only (the forgetting probe)
        m0 = (yte == 0) | (yte == 1)
        pred0 = net.predict(Xte[m0], steps)
        a0 = (pred0 == yte[m0]).float().mean().item()
        acc_task0.append(a0)
    # final accuracy over ALL classes
    correct = sum((net.predict(Xte[b:b+500], steps) == yte[b:b+500]).sum().item()
                  for b in range(0, len(Xte), 500)) / len(Xte)
    return acc_task0, correct


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--hidden", type=int, default=500)
    ap.add_argument("--steps", type=int, default=25)
    ap.add_argument("--beta", type=float, default=0.5)
    ap.add_argument("--n-train", type=int, default=60000)
    ap.add_argument("--n-test", type=int, default=10000)
    args = ap.parse_args()

    dev = pick_device(); torch.manual_seed(0)
    print(f"device: {dev} | class-incremental forgetting (EP): full vs frozen-features")
    tf = transforms.ToTensor()
    tr = datasets.MNIST("./data", train=True, download=True, transform=tf)
    te = datasets.MNIST("./data", train=False, download=True, transform=tf)
    Xtr = tr.data[:args.n_train].float().view(-1, 784).to(dev)/255.0; ytr = tr.targets[:args.n_train].to(dev)
    Xte = te.data[:args.n_test].float().view(-1, 784).to(dev)/255.0; yte = te.targets[:args.n_test].to(dev)

    for arm in ("full", "frozen"):
        t0 = time.time()
        a0, final = run(arm, Xtr, ytr, Xte, yte, args.hidden, args.epochs, args.batch,
                        args.beta, args.steps, dev)
        stages = "  ".join(f"{a*100:4.0f}" for a in a0)
        print(f"\n[{arm}]  task-0 acc after each stage:  {stages}   ({time.time()-t0:.0f}s)")
        print(f"        task-0: {a0[0]*100:.0f}% -> {a0[-1]*100:.0f}%  "
              f"(forgetting {(a0[0]-a0[-1])*100:+.0f} pp)   final all-class acc {final*100:.1f}%")


if __name__ == "__main__":
    main()
