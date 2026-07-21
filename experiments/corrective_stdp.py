"""Corrective STDP — give STDP the yes/no signal it lacks, keeping a pure local rule (Exp 14).

Exp 10 finding (ours): unsupervised STDP can't be steered — no error signal. FF's fix was a
positive/negative CONTRAST. This marries the two: FF's contrast, but the weight update is a pure
local Hebbian correlation (pre × post), NOT a goodness gradient. Strengthen the connections active
for a POSITIVE input (image + correct label), weaken those active for a NEGATIVE one (wrong label).

  positive (correct label):  W += lr * (post_activity outer pre_input)   [Hebbian]
  negative (wrong label):    W -= lr * (post_activity outer pre_input)    [anti-Hebbian]

That is strictly local (each weight = its own two neurons' correlation) and no surrogate gradient
— more hardware-faithful than FF's goodness-gradient. Question: does it match FF (95.1%)?

Honesty: this is close to Contrastive Hebbian Learning (1991) / EP's family; the fresh angle is
the spiking version. Lit-scout before claiming novelty. Compares vs FF 95.1% and STDP 88.6%.

Usage: .venv/bin/python experiments/corrective_stdp.py --epochs 30
"""

import argparse
import time

import torch
import torch.nn.functional as F

import snntorch as snn
from snntorch import surrogate

from forward_forward import overlay, get_mnist, pick_device, LABEL_SCALE


class CorrectiveLayer:
    """Spiking layer, contrastive-Hebbian local update (no gradient). Weights row-normalised."""

    def __init__(self, n_in, n_out, T, lr, device, thresh=0.5, beta=0.9):
        self.W = (torch.randn(n_out, n_in, device=device) * (1.0 / n_in ** 0.5))
        self.W /= self.W.norm(dim=1, keepdim=True) + 1e-6
        self.T, self.lr, self.device = T, lr, device
        self.thresh, self.beta, self.n_out = thresh, beta, n_out

    @torch.no_grad()
    def activity(self, x):
        """Manual LIF (subtract reset), spike-count over T. No grad needed."""
        m = torch.zeros(x.size(0), self.n_out, device=self.device)
        drive = x @ self.W.t(); acc = 0.0
        for _ in range(self.T):
            m = self.beta * m + drive
            s = (m >= self.thresh).float()
            m = m - s * self.thresh
            acc = acc + s
        return acc                                    # [B, n_out] spike counts

    @torch.no_grad()
    def update(self, x_pos, x_neg):
        a_pos, a_neg = self.activity(x_pos), self.activity(x_neg)
        B = x_pos.size(0)
        # local contrastive-Hebbian: strengthen positive correlations, weaken negative
        dW = (a_pos.t() @ x_pos - a_neg.t() @ x_neg) / B
        self.W += self.lr * dW
        self.W /= self.W.norm(dim=1, keepdim=True) + 1e-6   # normalise (competition, stability)
        return a_pos.pow(2).mean().item(), a_neg.pow(2).mean().item()

    @torch.no_grad()
    def norm_forward(self, x):
        a = self.activity(x)
        return a / (a.norm(dim=1, keepdim=True) + 1e-6)

    def goodness(self, x):
        return self.activity(x).pow(2).mean(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--hidden", type=int, default=500)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--T", type=int, default=12)
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--n-train", type=int, default=60000)
    ap.add_argument("--n-test", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    dev = pick_device(); torch.manual_seed(args.seed)
    print(f"device: {dev} | Corrective STDP (contrastive-Hebbian) | {args.layers}x{args.hidden} T={args.T}")

    tr, te = get_mnist(args.batch)
    Xtr = ((tr.data[:args.n_train].float().view(-1, 784)/255.0)-0.1307)/0.3081; ytr = tr.targets[:args.n_train]
    Xte = ((te.data[:args.n_test].float().view(-1, 784)/255.0)-0.1307)/0.3081; yte = te.targets[:args.n_test]

    dims = [784+10] + [args.hidden]*args.layers
    layers = [CorrectiveLayer(dims[i], dims[i+1], args.T, args.lr, dev) for i in range(args.layers)]

    for ep in range(1, args.epochs+1):
        t0 = time.time(); perm = torch.randperm(len(Xtr))
        for b in range(0, len(Xtr), args.batch):
            idx = perm[b:b+args.batch]
            x, y = Xtr[idx].to(dev), ytr[idx].to(dev)
            y_neg = (y + torch.randint(1, 10, y.shape, device=dev)) % 10
            ip, in_ = overlay(x, y), overlay(x, y_neg)
            for L in layers:
                L.update(ip, in_)
                ip, in_ = L.norm_forward(ip), L.norm_forward(in_)
        if ep % 5 == 0 or ep == 1:
            correct = 0
            for b in range(0, len(Xte), 512):
                x = Xte[b:b+512].to(dev)
                scores = torch.zeros(x.size(0), 10, device=dev)
                for lab in range(10):
                    inp = overlay(x, torch.full((x.size(0),), lab, device=dev))
                    tot = 0.0
                    for L in layers:
                        tot = tot + L.goodness(inp); inp = L.norm_forward(inp)
                    scores[:, lab] = tot
                correct += (scores.argmax(1).cpu() == yte[b:b+512]).sum().item()
            print(f"  ep {ep:2d}  test {correct/len(Xte)*100:5.2f}%  ({time.time()-t0:.0f}s)")

    print("\n(vs FF 95.1%, STDP 88.6%; EP 97.6% | BP ref 98.9%)")


if __name__ == "__main__":
    main()
