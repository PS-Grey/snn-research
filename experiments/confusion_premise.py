"""Step 0 — does confusion structure hold in EP, stably, and is it self-readable? (premise check)

We already know (old regime: hybrid + ensembles, vault 'Architecture Reference — Plain English') that
per-class difficulty/confusion EMERGES on its own. This checks only the gap that finding leaves for
the CURRENT backprop-free EP net, before we build any confusion-aware method:

  1. concentrated?  -- is error mass piled on a few pairs (7-1, 4-9...) or smeared evenly?
  2. stable?        -- are the same hot pairs hot on every seed? (else it is noise, cf. Exp 18/21)
  3. self-readable?  -- when the net sees class c, is its RUNNER-UP (2nd-strongest output) the class
                        it actually confuses c with? If yes, the net can build its own confusion map
                        locally, on-chip, with no labels -- the signal a confusion-aware rule needs.

No new method. Just measurement. Multi-seed from line one (our standing rule).
Usage: .venv/bin/python experiments/confusion_premise.py --seeds 3 --epochs 10
"""

import argparse
import time

import torch
import torch.nn.functional as F
from torchvision import datasets, transforms

from equilibrium_prop import EPNet, pick_device, rho


@torch.no_grad()
def outputs(net, X, steps, batch=500):
    ys = [rho(net.settle(X[b:b+batch], None, 0.0, steps)[1]) for b in range(0, len(X), batch)]
    return torch.cat(ys)


def train_seed(seed, Xtr, ytr, epochs, batch, beta, steps, hidden, dev):
    net = EPNet(hidden, dev, n_classes=10, seed=seed)
    for _ in range(epochs):
        perm = torch.randperm(len(Xtr), device=dev)
        for b in range(0, len(Xtr), batch):
            idx = perm[b:b+batch]
            net.train_step(Xtr[idx], F.one_hot(ytr[idx], 10).float(), beta, steps)
    return net


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--hidden", type=int, default=500)
    ap.add_argument("--steps", type=int, default=25)
    ap.add_argument("--beta", type=float, default=0.5)
    ap.add_argument("--n-train", type=int, default=20000)
    args = ap.parse_args()

    dev = pick_device()
    tf = transforms.ToTensor()
    tr = datasets.MNIST("./data", train=True, download=True, transform=tf)
    te = datasets.MNIST("./data", train=False, download=True, transform=tf)
    Xtr = tr.data[:args.n_train].float().view(-1, 784).to(dev)/255.0; ytr = tr.targets[:args.n_train].to(dev)
    Xte = te.data.float().view(-1, 784).to(dev)/255.0; yte = te.targets.to(dev)
    print(f"device: {dev} | confusion premise check (EP, MNIST) | {args.seeds} seeds")

    confs, runnermatch, accs = [], [], []
    for seed in range(args.seeds):
        t0 = time.time()
        net = train_seed(seed, Xtr, ytr, args.epochs, args.batch, args.beta, args.steps, args.hidden, dev)
        Y = outputs(net, Xte, args.steps)
        pred = Y.argmax(1)
        acc = (pred == yte).float().mean().item(); accs.append(acc)
        # confusion matrix (rows = true, cols = predicted), off-diagonal only
        C = torch.zeros(10, 10)
        for t, p in zip(yte.tolist(), pred.tolist()):
            C[t, p] += 1
        off = C.clone(); off.fill_diagonal_(0)
        confs.append(off)
        # runner-up per true class: 2nd-highest output, its mode; vs the class's top error target
        rmatch = 0
        for c in range(10):
            m = yte == c
            second = Y[m].topk(2, dim=1).indices[:, 1]           # runner-up class for every sample of c
            top_runner = torch.bincount(second, minlength=10).argmax().item()
            top_error = off[c].argmax().item()                   # class c is most often mislabelled as
            rmatch += (top_runner == top_error)
        runnermatch.append(rmatch / 10)
        # this seed's hottest confusion pairs
        flat = off.flatten(); top = flat.topk(5).indices
        pairs = [(int(i // 10), int(i % 10), int(flat[i])) for i in top]
        print(f"  seed {seed}: acc {acc*100:.1f}%  hot pairs " +
              " ".join(f"{a}->{b}({n})" for a, b, n in pairs) +
              f"  runner=error {runnermatch[-1]*100:.0f}%  ({time.time()-t0:.0f}s)")

    # concentration: share of all errors sitting in the top-5 pairs (uniform baseline = 5/90 = 5.6%)
    concentr = [off.flatten().topk(5).values.sum().item() / off.sum().item() for off in confs]
    # stability: correlation of the off-diagonal confusion matrices across seeds
    import itertools
    corrs = []
    for i, j in itertools.combinations(range(args.seeds), 2):
        a, b = confs[i].flatten(), confs[j].flatten()
        corrs.append(torch.corrcoef(torch.stack([a, b]))[0, 1].item())
    print(f"\nacc {sum(accs)/len(accs)*100:.1f}%")
    print(f"1. CONCENTRATED  top-5 pairs hold {sum(concentr)/len(concentr)*100:.0f}% of all errors "
          f"(uniform would be ~6%)")
    print(f"2. STABLE        cross-seed confusion-matrix correlation {sum(corrs)/len(corrs):.2f} "
          f"(1.0 = identical structure)" if corrs else "2. STABLE  (need >=2 seeds)")
    print(f"3. SELF-READABLE runner-up = actual-error-target for "
          f"{sum(runnermatch)/len(runnermatch)*100:.0f}% of classes on average")
    print("\nverdict: premise holds in EP if all three are high (concentrated, correlated, runner-up matches).")


if __name__ == "__main__":
    main()
