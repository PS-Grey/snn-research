"""Learned graded-spike PAYLOAD — the untouched open problem (scaffold, Exp 20).

Scout-confirmed open (2026-07-22, vault [[SNN Novelty Target ...]]): a plastic, separately-learned
per-spike graded PAYLOAD -- a second information channel with its own rule -- shown to add capacity
AT MATCHED SPIKE BUDGET. Loihi 2 already has integer graded spikes in silicon; what's missing is a
*learned* payload that earns its bits.

Claim under test:
    a per-spike graded payload, learned by its own input-dependent projection, adds task info at
    matched spike budget -- beyond a binary/rate code AND beyond a *computed* (membrane) payload.

Design that makes "matched spike budget" exact: the payload rides on spikes that already fire, so
it adds ZERO extra spikes. Each hidden spike j transmits  s_j * payload_j  (not just s_j); the
readout W2 is identical across modes. Only the payload differs:
    binary   -- payload = 1                         (rate code baseline)
    computed -- payload = membrane u_j at spike      (sigma-delta; the KNOWN method)
    learned  -- payload = softplus(V1_j . x)         (separate learned projection; the NOVEL channel)

This scaffold trains with a surrogate gradient -- a PROBE for whether the payload channel carries
capacity at all. If it does, the real target is a LOCAL Delta-payload rule (on-chip); if it does
not, the idea dies cheap. HONEST CONTROL still owed before any novelty claim: param-match the binary
baseline (learned mode has extra params V1), else we only show "more params help". See --wide.

Usage: .venv/bin/python experiments/graded_payload.py --epochs 4
"""

import argparse
import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms


def pick_device():
    return (torch.device("mps") if torch.backends.mps.is_available()
            else torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))


class SpikeFn(torch.autograd.Function):
    """Binary threshold with atan surrogate gradient (input is membrane - threshold)."""
    @staticmethod
    def forward(ctx, u):
        ctx.save_for_backward(u)
        return (u > 0).float()

    @staticmethod
    def backward(ctx, g):
        (u,) = ctx.saved_tensors
        return g / (1.0 + (math.pi * u) ** 2)          # alpha folded into pi


spike = SpikeFn.apply


class GradedNet(nn.Module):
    """784 -> N LIF (graded spikes) -> 10. The payload rescales each spike; spikes are unchanged
    by the payload mode, so spike budget is matched across modes by construction."""

    def __init__(self, hidden, mode, beta=0.9, wide=1):
        super().__init__()
        self.mode, self.beta, self.hidden = mode, beta, hidden
        h = hidden * wide                              # --wide grows the binary baseline for param-match
        self.fc1 = nn.Linear(784, h)
        self.bn1 = nn.BatchNorm1d(h)
        self.fc2 = nn.Linear(h, 10)
        if mode == "learned":                          # learned payload = second projection of x
            self.pay = nn.Linear(784, h)
        elif mode == "learned_u":                      # learned READOUT of the temporal membrane state
            self.pay = nn.Linear(h, h)                 #   (orthogonal to the rate code; learns > raw u?)

    def payload(self, u, x):
        if self.mode == "binary":
            return 1.0
        if self.mode == "computed":
            return u.clamp(min=0.0)                    # membrane at spike = sigma-delta graded value
        if self.mode == "learned":
            return F.softplus(self.pay(x))             # redundant with rate code (probe showed: no gain)
        return F.softplus(self.pay(u.clamp(min=0.0)))  # learned mix of population membrane = orthogonal

    def forward(self, x, T):
        u = torch.zeros(x.size(0), self.fc1.out_features, device=x.device)
        out = torch.zeros(x.size(0), 10, device=x.device)
        nspk = 0.0
        cur0 = self.bn1(self.fc1(x))                    # rate-coded input -> constant drive per step
        for _ in range(T):
            u = self.beta * u + cur0
            s = spike(u - 1.0)                          # threshold 1
            u = u - s                                   # subtract reset
            v = s * self.payload(u, x)                  # graded spike value (matched spikes)
            out = out + self.fc2(v)
            nspk = nspk + s.sum().item()
        return out / T, nspk / x.size(0)                # logits, spikes per image


def get_mnist(n_train, n_test, dev):
    tf = transforms.ToTensor()
    tr = datasets.MNIST("./data", train=True, download=True, transform=tf)
    te = datasets.MNIST("./data", train=False, download=True, transform=tf)
    Xtr = tr.data[:n_train].float().view(-1, 784).to(dev) / 255.0; ytr = tr.targets[:n_train].to(dev)
    Xte = te.data[:n_test].float().view(-1, 784).to(dev) / 255.0; yte = te.targets[:n_test].to(dev)
    return Xtr, ytr, Xte, yte


def run(mode, wide, Xtr, ytr, Xte, yte, hidden, T, epochs, batch, lr, dev):
    net = GradedNet(hidden, mode, wide=wide).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    for _ in range(epochs):
        net.train(); perm = torch.randperm(len(Xtr), device=dev)
        for b in range(0, len(Xtr), batch):
            idx = perm[b:b+batch]
            logits, _ = net(Xtr[idx], T)
            loss = F.cross_entropy(logits, ytr[idx])
            opt.zero_grad(); loss.backward(); opt.step()
    net.eval(); correct = 0; spk = 0.0; nb = 0
    with torch.no_grad():
        for b in range(0, len(Xte), 500):
            logits, s = net(Xte[b:b+500], T)
            correct += (logits.argmax(1) == yte[b:b+500]).sum().item(); spk += s; nb += 1
    nparam = sum(p.numel() for p in net.parameters())
    return correct / len(Xte), spk / nb, nparam


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--hidden", type=int, default=300)
    ap.add_argument("--T", type=int, default=15)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--n-train", type=int, default=20000)
    ap.add_argument("--n-test", type=int, default=10000)
    args = ap.parse_args()

    dev = pick_device(); torch.manual_seed(0)
    print(f"device: {dev} | graded-payload PROBE (surrogate grad) | hidden {args.hidden}, T {args.T}")
    Xtr, ytr, Xte, yte = get_mnist(args.n_train, args.n_test, dev)

    print(f"\n{'mode':12s} {'test acc':>9s} {'spk/img':>8s} {'acc/kspk':>9s} {'params':>8s}")
    for mode in ("binary", "computed", "learned", "learned_u"):
        t0 = time.time()
        acc, spk, npar = run(mode, 1, Xtr, ytr, Xte, yte, args.hidden, args.T,
                             args.epochs, args.batch, args.lr, dev)
        print(f"{mode:12s} {acc*100:8.2f}% {spk:8.0f} {acc/(spk/1000):9.2f} {npar:8d}  ({time.time()-t0:.0f}s)")
    print("\nprobe only. next if capacity shows: (1) param-matched binary control (--wide), "
          "(2) swap surrogate grad -> local Delta-payload rule (on-chip target).")


if __name__ == "__main__":
    main()
