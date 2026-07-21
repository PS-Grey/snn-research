"""Spiking Forward-Forward — a backprop-free rule that trains the FEATURES (Exp 11).

Exp 10 hit the ceiling: unsupervised STDP features can't be improved by a global signal (Hebbian
reinforces errors; anti-Hebbian destabilises). The prescription was a *stable corrective* local
rule that trains features. Forward-Forward (Hinton 2022) is exactly that: each layer has its own
LOCAL objective (no backprop between layers) -- fire HARD ('goodness' high) for a POSITIVE input
(image + correct label) and WEAKLY for a NEGATIVE input (image + wrong label). Corrective
(positive vs negative = signed), and it trains every layer, not just a readout.

Spiking form: layer activity = LIF spike counts over T; goodness = mean(activity^2); the layer's
own surrogate gradient does the LOCAL update (single-layer, no chain through the net). Activity is
L2-normalised before the next layer so it learns the pattern, not the magnitude (which IS the
goodness). Inference: overlay each of the 10 labels, run the net, predict the label with the
highest summed goodness.

Compares to the STDP + surprise-readout line (our 91.27% backprop-free best). No global backprop.

Usage: .venv/bin/python experiments/forward_forward.py --epochs 30
"""

import argparse
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

import snntorch as snn
from snntorch import surrogate


def pick_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


LABEL_SCALE = 10.0   # amplify the one-hot so the label is not drowned by the 784 image dims


def overlay(x, y, n_cls=10):
    """Concatenate the image with an amplified one-hot label block -> FF input [B, 784+10]."""
    oh = F.one_hot(y, n_cls).float() * LABEL_SCALE
    return torch.cat([x, oh], dim=1)


class FFLayer(nn.Module):
    """One spiking layer with its OWN local Forward-Forward objective (no backprop between layers)."""

    def __init__(self, n_in, n_out, T=15, beta=0.9, thresh=2.0, lr=3e-3):
        super().__init__()
        self.fc = nn.Linear(n_in, n_out)
        # NO BatchNorm: FF's signal IS activity magnitude (goodness); BN would normalise it away.
        self.lif = snn.Leaky(beta=beta, spike_grad=surrogate.atan(), reset_mechanism="subtract")
        self.T, self.thresh = T, thresh
        self.opt = torch.optim.Adam(self.parameters(), lr=lr)

    def activity(self, x):
        """Spike-count activity per neuron over T timesteps (graded input coding)."""
        m = self.lif.init_leaky()
        drive = self.fc(x)                       # constant input current across the T steps
        acc = 0.0
        for _ in range(self.T):
            s, m = self.lif(drive, m)
            acc = acc + s
        return acc                               # [B, n_out]

    def goodness(self, x):
        return self.activity(x).pow(2).mean(1)   # scalar per sample

    def train_step(self, x_pos, x_neg):
        """Local FF update. Adaptive threshold = running-mean goodness, so the loss pushes pos
        ABOVE and neg BELOW the current mean -> forces SEPARATION, not collapse to a fixed value
        (a fixed thresh far from the operating goodness only pushes one side -> trivial collapse)."""
        g_pos, g_neg = self.goodness(x_pos), self.goodness(x_neg)
        thr = 0.5 * (g_pos.mean() + g_neg.mean()).detach()
        loss = (F.softplus(thr - g_pos) + F.softplus(g_neg - thr)).mean()
        self.opt.zero_grad(); loss.backward(); self.opt.step()
        return loss.item(), g_pos.mean().item(), g_neg.mean().item()

    @torch.no_grad()
    def forward_norm(self, x):
        """Detached, L2-normalised activity to feed the next layer (magnitude carries goodness)."""
        a = self.activity(x)
        return (a / (a.norm(dim=1, keepdim=True) + 1e-6)).detach()


def get_mnist(batch):
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    tr = datasets.MNIST("./data", train=True, download=True, transform=tf)
    te = datasets.MNIST("./data", train=False, download=True, transform=tf)
    return tr, te


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--hidden", type=int, default=500)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--T", type=int, default=15)
    ap.add_argument("--n-train", type=int, default=60000)
    ap.add_argument("--n-test", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    dev = pick_device(); torch.manual_seed(args.seed)
    print(f"device: {dev} | spiking Forward-Forward | {args.layers}x{args.hidden}, T={args.T}")

    tr, te = get_mnist(args.batch)
    Xtr = tr.data[:args.n_train].float().view(-1, 784) / 255.0
    Xtr = (Xtr - 0.1307) / 0.3081
    ytr = tr.targets[:args.n_train]
    Xte = te.data[:args.n_test].float().view(-1, 784) / 255.0
    Xte = (Xte - 0.1307) / 0.3081
    yte = te.targets[:args.n_test]

    dims = [784 + 10] + [args.hidden] * args.layers
    layers = [FFLayer(dims[i], dims[i+1], T=args.T).to(dev) for i in range(args.layers)]

    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        perm = torch.randperm(len(Xtr))
        gp = gn = 0.0; nb = 0
        for b in range(0, len(Xtr), args.batch):
            idx = perm[b:b+args.batch]
            x, y = Xtr[idx].to(dev), ytr[idx].to(dev)
            y_neg = (y + torch.randint(1, 10, y.shape, device=dev)) % 10   # wrong label
            inp_pos = overlay(x, y); inp_neg = overlay(x, y_neg)
            # train each layer locally, feeding the previous layer's normalised activity forward
            for L in layers:
                _, p, n = L.train_step(inp_pos, inp_neg)
                inp_pos, inp_neg = L.forward_norm(inp_pos), L.forward_norm(inp_neg)
                gp += p; gn += n; nb += 1
        if ep % 5 == 0 or ep == 1:
            # eval: predict the label whose overlay gives the highest total goodness
            correct = 0
            with torch.no_grad():
                for b in range(0, len(Xte), 512):
                    x = Xte[b:b+512].to(dev)
                    scores = torch.zeros(x.size(0), 10, device=dev)
                    for lab in range(10):
                        inp = overlay(x, torch.full((x.size(0),), lab, device=dev))
                        tot = 0.0
                        for L in layers:
                            tot = tot + L.goodness(inp)
                            inp = L.forward_norm(inp)
                        scores[:, lab] = tot
                    correct += (scores.argmax(1).cpu() == yte[b:b+512]).sum().item()
            print(f"  ep {ep:2d}  test {correct/len(Xte)*100:5.2f}%  "
                  f"g_pos {gp/nb:.2f} g_neg {gn/nb:.2f}  ({time.time()-t0:.0f}s)")

    print(f"\nbackprop-free reference line: pure STDP 88.6% -> surprise-readout 91.27% "
          f"-> FF (this) ?  | surrogate-gradient 98.9%")


if __name__ == "__main__":
    main()
