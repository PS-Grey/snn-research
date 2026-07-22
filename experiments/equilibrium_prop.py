"""Equilibrium Propagation — the most neuromorphic-ready backprop-free rule (Exp 13).

Why EP: its weight update is STRICTLY two-neuron-local -- each synapse changes from only the two
neurons it connects, measured in two settling states. That is exactly what a physical chip synapse
can do (it sees only its pre/post neuron), so EP is the local rule with the clearest path to
on-chip learning. Backprop needs global error routed backward; EP needs none.

Mechanism (Scellier & Bengio 2017): an energy-based net settles to equilibrium.
  Free phase   (beta=0): clamp input, let hidden+output settle. Record states.
  Nudged phase (beta>0): also pull the output gently toward the target; settle again.
  Update: dW ~ (1/beta)[ rho(s_i)rho(s_j) |_nudged - rho(s_i)rho(s_j) |_free ]  -- purely local.
Symmetric nudging (+beta and -beta, Laborieux 2021) cancels the finite-difference bias.

Canonical EP is RATE-based (analog neurons), not spiking -- its neuromorphic appeal is the local
rule + analog-hardware fit. Spiking-EP is the follow-up. MNIST here; CIFAR version -> Colab.

Usage: .venv/bin/python experiments/equilibrium_prop.py --epochs 20
"""

import argparse
import time

import torch
import torch.nn.functional as F
from torchvision import datasets, transforms


def pick_device():
    return (torch.device("mps") if torch.backends.mps.is_available()
            else torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))


def rho(x):
    return x.clamp(0, 1)            # hard-sigmoid nonlinearity


def rho_p(x):
    return ((x >= 0) & (x <= 1)).float()


class EPNet:
    """784 -> hidden -> 10, symmetric weights, energy-based settling. Local EP updates."""

    def __init__(self, hidden, device, lr=(0.1, 0.05), seed=0):
        g = torch.Generator().manual_seed(seed)
        k1 = (2 / 784) ** 0.5; k2 = (2 / hidden) ** 0.5
        self.W1 = (torch.randn(hidden, 784, generator=g) * k1).to(device)   # x <-> h
        self.W2 = (torch.randn(10, hidden, generator=g) * k2).to(device)    # h <-> y
        self.bh = torch.zeros(hidden, device=device)
        self.by = torch.zeros(10, device=device)
        self.hidden, self.device, self.lr = hidden, device, lr

    def settle(self, x, y_target, beta, steps, dt=0.5):
        """Relax hidden h and output y to equilibrium. beta>0 nudges y toward the target."""
        B = x.size(0)
        h = torch.zeros(B, self.hidden, device=self.device)
        y = torch.zeros(B, 10, device=self.device)
        rx = rho(x)
        for _ in range(steps):
            # each neuron's drive = its connected neurons (two-neuron locality, both directions)
            h_in = rx @ self.W1.t() + rho(y) @ self.W2 + self.bh
            y_in = rho(h) @ self.W2.t() + self.by
            h = (h + dt * (-h + rho_p(h) * h_in)).clamp(0, 1)
            dy = -y + rho_p(y) * y_in
            if beta != 0:
                dy = dy + beta * (y_target - y)          # nudge output toward target
            y = (y + dt * dy).clamp(0, 1)
        return h, y

    def train_step(self, x, y_oh, beta, steps, freeze_feat=False):
        h0, y0 = self.settle(x, None, 0.0, steps)                 # free phase
        hp, yp = self.settle(x, y_oh, beta, steps)                # +beta nudged
        hm, ym = self.settle(x, y_oh, -beta, steps)               # -beta nudged (symmetric)
        rx = rho(x)
        lr1, lr2 = self.lr
        # local EP update: change in two-neuron correlation between the nudged states
        c = 1.0 / (2 * beta * x.size(0))
        self.W2 += lr2 * c * (rho(yp).t() @ rho(hp) - rho(ym).t() @ rho(hm))
        self.by += lr2 * c * (rho(yp) - rho(ym)).sum(0)
        if not freeze_feat:                                       # freeze the FEATURE layer
            self.W1 += lr1 * c * (rho(hp).t() @ rx - rho(hm).t() @ rx)
            self.bh += lr1 * c * (rho(hp) - rho(hm)).sum(0)
        return (y0.argmax(1) == y_oh.argmax(1)).float().mean().item()

    @torch.no_grad()
    def predict(self, x, steps):
        _, y = self.settle(x, None, 0.0, steps)
        return y.argmax(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--hidden", type=int, default=500)
    ap.add_argument("--steps", type=int, default=25)
    ap.add_argument("--beta", type=float, default=0.5)
    ap.add_argument("--n-train", type=int, default=60000)
    ap.add_argument("--n-test", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    dev = pick_device(); torch.manual_seed(args.seed)
    print(f"device: {dev} | Equilibrium Prop | hidden {args.hidden}, settle {args.steps}, beta {args.beta}")

    tf = transforms.ToTensor()
    tr = datasets.MNIST("./data", train=True, download=True, transform=tf)
    te = datasets.MNIST("./data", train=False, download=True, transform=tf)
    Xtr = tr.data[:args.n_train].float().view(-1, 784).to(dev) / 255.0
    ytr = tr.targets[:args.n_train].to(dev)
    Xte = te.data[:args.n_test].float().view(-1, 784).to(dev) / 255.0
    yte = te.targets[:args.n_test].to(dev)

    net = EPNet(args.hidden, dev, seed=args.seed)
    for ep in range(1, args.epochs + 1):
        t0 = time.time(); perm = torch.randperm(len(Xtr), device=dev); acc = 0.0; nb = 0
        for b in range(0, len(Xtr), args.batch):
            idx = perm[b:b+args.batch]
            x, y = Xtr[idx], F.one_hot(ytr[idx], 10).float()
            acc += net.train_step(x, y, args.beta, args.steps); nb += 1
        if ep % 2 == 0 or ep == 1:
            correct = sum((net.predict(Xte[b:b+500], args.steps) == yte[b:b+500]).sum().item()
                          for b in range(0, len(Xte), 500))
            print(f"  ep {ep:2d}  train {acc/nb*100:5.2f}%  test {correct/len(Xte)*100:5.2f}%"
                  f"  ({time.time()-t0:.0f}s)")

    print("\nbackprop-free line: STDP 88.6 -> surprise 91.3 -> FF 95.1 -> EP (this) ? | BP ref 98.9")


if __name__ == "__main__":
    main()
