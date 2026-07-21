"""Recurrent / top-down spiking Forward-Forward — the 'FFF' idea (Exp 12).

Sergiy's question: if two forward passes give a yes/no contrast, do MORE passes that refine each
other help? That iterative settling is the bridge from Forward-Forward to predictive coding.

Feedforward FF (Exp 11): layers run one-after-another. Here both layers run inside one T-step loop
and layer 2 feeds BACK to layer 1 each step, so the representation SETTLES over the T timesteps
(layer 1's guess refines layer 2's, which refines layer 1's, ...). Each layer still trains only on
its own local goodness — the top-down weight is just another of layer 1's inputs; cross-layer
signals are detached, so no backprop between layers. Local, corrective, backprop-free.

Clean A/B on the same net: --topdown (settling) vs --no-topdown (feedforward-in-a-loop baseline).
Does the top-down feedback beat plain FF's 95.1%?

Usage: .venv/bin/python experiments/forward_forward_recurrent.py --epochs 40 --topdown
"""

import argparse
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

import snntorch as snn
from snntorch import surrogate

from forward_forward import overlay, get_mnist, pick_device, LABEL_SCALE


def ff_loss(g_pos, g_neg):
    """Adaptive-threshold FF loss: push pos above, neg below the running-mean goodness."""
    thr = 0.5 * (g_pos.mean() + g_neg.mean()).detach()
    return (F.softplus(thr - g_pos) + F.softplus(g_neg - thr)).mean()


class RecurrentFF(nn.Module):
    def __init__(self, n_in, hidden, T=15, beta=0.9, topdown=True):
        super().__init__()
        self.ff1 = nn.Linear(n_in, hidden)      # input -> L1  (bottom-up)
        self.ff2 = nn.Linear(hidden, hidden)    # L1 -> L2      (bottom-up)
        self.td = nn.Linear(hidden, hidden)     # L2 -> L1      (top-down feedback)
        mk = lambda: snn.Leaky(beta=beta, spike_grad=surrogate.atan(), reset_mechanism="subtract")
        self.lif1, self.lif2 = mk(), mk()
        self.T, self.topdown = T, topdown

    def settle(self, x):
        """Run the T-step settling. Returns (goodness_L1, goodness_L2). Cross-layer signals are
        detached so each layer's gradient touches only its own weights (FF locality)."""
        m1 = self.lif1.init_leaky(); m2 = self.lif2.init_leaky()
        s2_prev = torch.zeros(x.size(0), self.ff2.out_features, device=x.device)
        acc1 = acc2 = 0.0
        base = self.ff1(x)                      # bottom-up drive (constant across steps)
        for _ in range(self.T):
            drive1 = base + (self.td(s2_prev.detach()) if self.topdown else 0.0)
            s1, m1 = self.lif1(drive1, m1)       # grad -> ff1, td
            s2, m2 = self.lif2(self.ff2(s1.detach()), m2)  # grad -> ff2 only
            acc1 = acc1 + s1; acc2 = acc2 + s2
            s2_prev = s2
        return acc1.pow(2).mean(1), acc2.pow(2).mean(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--hidden", type=int, default=500)
    ap.add_argument("--T", type=int, default=15)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--topdown", action="store_true")
    ap.add_argument("--no-topdown", dest="topdown", action="store_false")
    ap.add_argument("--n-train", type=int, default=60000)
    ap.add_argument("--n-test", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.set_defaults(topdown=True)
    args = ap.parse_args()

    dev = pick_device(); torch.manual_seed(args.seed)
    print(f"device: {dev} | recurrent FF | top-down={args.topdown} | {args.hidden}h T={args.T}")

    tr, te = get_mnist(args.batch)
    Xtr = ((tr.data[:args.n_train].float().view(-1, 784)/255.0)-0.1307)/0.3081; ytr = tr.targets[:args.n_train]
    Xte = ((te.data[:args.n_test].float().view(-1, 784)/255.0)-0.1307)/0.3081; yte = te.targets[:args.n_test]

    net = RecurrentFF(784+10, args.hidden, T=args.T, topdown=args.topdown).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)

    for ep in range(1, args.epochs+1):
        t0 = time.time(); perm = torch.randperm(len(Xtr))
        for b in range(0, len(Xtr), args.batch):
            idx = perm[b:b+args.batch]
            x, y = Xtr[idx].to(dev), ytr[idx].to(dev)
            y_neg = (y + torch.randint(1, 10, y.shape, device=dev)) % 10
            gp1, gp2 = net.settle(overlay(x, y))
            gn1, gn2 = net.settle(overlay(x, y_neg))
            loss = ff_loss(gp1, gn1) + ff_loss(gp2, gn2)   # each layer's own local FF loss
            opt.zero_grad(); loss.backward(); opt.step()
        if ep % 5 == 0 or ep == 1:
            correct = 0
            with torch.no_grad():
                for b in range(0, len(Xte), 512):
                    x = Xte[b:b+512].to(dev)
                    scores = torch.zeros(x.size(0), 10, device=dev)
                    for lab in range(10):
                        g1, g2 = net.settle(overlay(x, torch.full((x.size(0),), lab, device=dev)))
                        scores[:, lab] = g1 + g2
                    correct += (scores.argmax(1).cpu() == yte[b:b+512]).sum().item()
            print(f"  ep {ep:2d}  test {correct/len(Xte)*100:5.2f}%  ({time.time()-t0:.0f}s)")

    print(f"\n(feedforward FF baseline = 95.1%; surrogate-gradient ref 98.9%)")


if __name__ == "__main__":
    main()
