"""Battery of self-signals — can ANY local, label-free signal read the confusion? (Exp 24)

Exp 23 killed ONE signal (the output runner-up, r=-0.18). That does not prove no signal exists (the
confusion is systematic, so the info is IN the network). This tests a battery, asking of each: does
its off-diagonal predict WHERE the errors actually go (the true confusion matrix)?

  runner-up   -- output layer, 2nd-strongest class histogram      (Exp 23's failed signal, baseline)
  mean-output -- output layer, soft leakage: how much class c's samples activate output d
  hidden-sim  -- HIDDEN layer overlap: cosine similarity of class means. This is where confusion
                 actually lives (shared feature-neurons) and it is co-firing, which a LOCAL rule can
                 see. The candidate that could revive the on-chip idea.
Plus a difficulty-LEVEL check: per-class output margin vs per-class error rate (can it tell WHEN it
struggles, even if not toward-what). Multi-seed. Success = a signal whose off-diagonal correlates
with the true confusion off-diagonal.

Usage: .venv/bin/python experiments/confusion_signals.py --seeds 3 --epochs 12
"""

import argparse
import time

import torch
import torch.nn.functional as F
from torchvision import datasets, transforms

from equilibrium_prop import EPNet, pick_device, rho


@torch.no_grad()
def feats(net, X, steps, batch=500):
    H, Y = [], []
    for b in range(0, len(X), batch):
        h, y = net.settle(X[b:b+batch], None, 0.0, steps)
        H.append(rho(h)); Y.append(rho(y))
    return torch.cat(H).cpu(), torch.cat(Y).cpu()


def offcorr(sig, conf):
    """Pearson correlation of the two matrices over off-diagonal entries only."""
    mask = ~torch.eye(10, dtype=torch.bool)
    a, b = sig[mask], conf[mask]
    if a.std() == 0 or b.std() == 0:
        return 0.0
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--hidden", type=int, default=500)
    ap.add_argument("--steps", type=int, default=25)
    ap.add_argument("--beta", type=float, default=0.5)
    ap.add_argument("--n-train", type=int, default=60000)
    args = ap.parse_args()

    dev = pick_device()
    tf = transforms.ToTensor()
    tr = datasets.MNIST("./data", train=True, download=True, transform=tf)
    te = datasets.MNIST("./data", train=False, download=True, transform=tf)
    Xtr = tr.data[:args.n_train].float().view(-1, 784).to(dev)/255.0; ytr = tr.targets[:args.n_train].to(dev)
    Xte = te.data.float().view(-1, 784).to(dev)/255.0; yte = te.targets.cpu()
    print(f"device: {dev} | confusion self-signal battery (EP, MNIST) | {args.seeds} seeds")

    sig_corr = {"runner-up": [], "mean-output": [], "hidden-sim": []}
    diff_corr = []
    for seed in range(args.seeds):
        t0 = time.time()
        net = EPNet(args.hidden, dev, n_classes=10, seed=seed)
        for _ in range(args.epochs):
            perm = torch.randperm(len(Xtr), device=dev)
            for b in range(0, len(Xtr), args.batch):
                idx = perm[b:b+args.batch]
                net.train_step(Xtr[idx], F.one_hot(ytr[idx], 10).float(), args.beta, args.steps)
        H, Y = feats(net, Xte, args.steps)
        pred = Y.argmax(1)

        conf = torch.zeros(10, 10)
        for t, p in zip(yte.tolist(), pred.tolist()):
            conf[t, p] += 1
        conf.fill_diagonal_(0)

        runner = torch.zeros(10, 10); meanout = torch.zeros(10, 10); meanH = torch.zeros(10, H.size(1))
        margin = torch.zeros(10); errrate = torch.zeros(10)
        for c in range(10):
            m = yte == c
            Yc, Hc = Y[m], H[m]
            second = Yc.topk(2, dim=1).indices[:, 1]
            runner[c] = torch.bincount(second, minlength=10).float()
            meanout[c] = Yc.mean(0)
            meanH[c] = Hc.mean(0)
            top2 = Yc.topk(2, dim=1).values
            margin[c] = (top2[:, 0] - top2[:, 1]).mean()
            errrate[c] = (Yc.argmax(1) != c).float().mean()
        # hidden overlap: cosine similarity between class-mean hidden vectors
        Hn = meanH / (meanH.norm(dim=1, keepdim=True) + 1e-8)
        hidsim = Hn @ Hn.t()

        sig_corr["runner-up"].append(offcorr(runner, conf))
        sig_corr["mean-output"].append(offcorr(meanout, conf))
        sig_corr["hidden-sim"].append(offcorr(hidsim, conf))
        # difficulty level: does low margin predict high error rate? (across the 10 classes)
        dc = torch.corrcoef(torch.stack([margin, errrate]))[0, 1].item()
        diff_corr.append(dc)
        print(f"  seed {seed}: acc {(pred==yte).float().mean()*100:.1f}%  "
              f"runner {sig_corr['runner-up'][-1]:+.2f}  meanout {sig_corr['mean-output'][-1]:+.2f}  "
              f"hidsim {sig_corr['hidden-sim'][-1]:+.2f}  ({time.time()-t0:.0f}s)")

    def mean(v): return sum(v)/len(v)
    print("\nHow well each self-signal predicts WHERE errors go (corr with true confusion, off-diag):")
    for k in ("runner-up", "mean-output", "hidden-sim"):
        print(f"  {k:12s} r = {mean(sig_corr[k]):+.2f}   [{' '.join(f'{x:+.2f}' for x in sig_corr[k])}]")
    print(f"\ndifficulty LEVEL: per-class margin vs error-rate r = {mean(diff_corr):+.2f} "
          f"(strong negative = the net knows WHEN it struggles, even if not toward-what)")
    print("\nverdict: a signal with |r| >~0.4 is a usable local read of confusion -> on-chip idea revives.")


if __name__ == "__main__":
    main()
