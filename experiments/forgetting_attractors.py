"""Is FORGETTING directional too — does it reuse the confusion attractors? (Exp 26)

Two findings so far look like they might be one mechanism:
  Exp 23/24: confusions are DIRECTIONAL — stuff falls INTO certain classes (9>3 but not 3>9). Some
             classes are magnetic attractors (deep, wide basins that swallow neighbours).
  Exp 15/16: forgetting is READOUT-SUPPRESSION — old classes get pushed toward zero by new learning.

Open question this raises (ours, not from a paper): when the net forgets, where does the forgotten
mass GO? If it collapses into the SAME classes that were already magnetic attractors, then
confusion-asymmetry and catastrophic forgetting are the same attractor-strength mechanism, and a
single quantity (basin dominance) explains both.

Protocol: train MNIST (0-9) to convergence -> measure attractor strength per class (how much error
mass falls INTO it). Then train Fashion (10-19) with NO replay -> severe forgetting. Then ask where
the now-misclassified MNIST samples land:
  Q1 concentration -- is the destination concentrated (few swallowing classes) or spread?
  Q2 new vs old    -- does forgotten mass go to the NEW classes, or to old magnetic ones?
  Q3 THE TEST      -- among OLD classes, does destination mass correlate with the attractor strength
                      measured BEFORE Fashion was ever seen? High r = one shared mechanism.

Usage: .venv/bin/python experiments/forgetting_attractors.py --seeds 3
"""

import argparse
import statistics as st
import time

import torch
import torch.nn.functional as F

from equilibrium_prop import EPNet, pick_device
from continual_hetero import NC, load_2020


@torch.no_grad()
def predict_all(net, X, steps, batch=500):
    return torch.cat([net.predict(X[b:b+batch], steps).cpu() for b in range(0, len(X), batch)])


def train_classes(net, Xtr, ytr, classes, epochs, batch, beta, steps, dev):
    m = torch.zeros_like(ytr, dtype=torch.bool)
    for c in classes:
        m |= (ytr == c)
    Xt, yt = Xtr[m], ytr[m]
    for _ in range(epochs):
        perm = torch.randperm(len(Xt), device=dev)
        for b in range(0, len(Xt), batch):
            idx = perm[b:b+batch]
            net.train_step(Xt[idx], F.one_hot(yt[idx], NC).float(), beta, steps)


def corr(a, b):
    if a.std() == 0 or b.std() == 0:
        return 0.0
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--hidden", type=int, default=500)
    ap.add_argument("--steps", type=int, default=25)
    ap.add_argument("--beta", type=float, default=0.5)
    ap.add_argument("--per-class", type=int, default=3000)
    args = ap.parse_args()

    dev = pick_device(); torch.manual_seed(0)
    print(f"device: {dev} | forgetting-attractor test | MNIST -> Fashion (no replay), {args.seeds} seeds")
    Xtr, ytr, Xte, yte = load_2020(args.per_class, dev)
    yte_c = yte.cpu()
    old_m = yte_c < 10                                     # MNIST test samples (the ones that get forgotten)

    concs, news, r_att, drops = [], [], [], []
    for seed in range(args.seeds):
        t0 = time.time()
        net = EPNet(args.hidden, dev, n_classes=NC, seed=seed)

        # ---- phase 1: learn MNIST, measure who is magnetic BEFORE Fashion exists
        train_classes(net, Xtr, ytr, list(range(10)), args.epochs, args.batch, args.beta, args.steps, dev)
        pred0 = predict_all(net, Xte, args.steps)
        acc0 = (pred0[old_m] == yte_c[old_m]).float().mean().item()
        C = torch.zeros(10, 10)
        for t, p in zip(yte_c[old_m].tolist(), pred0[old_m].tolist()):
            if p < 10:
                C[t, p] += 1
        C.fill_diagonal_(0)
        attractor = C.sum(0)                               # column sum = how much error mass falls INTO c

        # ---- phase 2: learn Fashion with NO replay -> catastrophic forgetting
        train_classes(net, Xtr, ytr, list(range(10, 20)), args.epochs, args.batch, args.beta, args.steps, dev)
        pred1 = predict_all(net, Xte, args.steps)
        acc1 = (pred1[old_m] == yte_c[old_m]).float().mean().item()

        # ---- phase 3: where did the forgotten MNIST mass go?
        wrong = old_m & (pred1 != yte_c)
        dest = torch.bincount(pred1[wrong], minlength=NC).float()
        conc = dest.topk(3).values.sum().item() / max(dest.sum().item(), 1)     # top-3 share
        new_share = dest[10:].sum().item() / max(dest.sum().item(), 1)
        r = corr(dest[:10], attractor)                     # THE test: same attractors as before?
        concs.append(conc); news.append(new_share); r_att.append(r); drops.append(acc0 - acc1)
        top = dest.topk(3).indices.tolist()
        print(f"  seed {seed}: MNIST {acc0*100:.1f}% -> {acc1*100:.1f}%  | top-3 sinks {top} "
              f"({conc*100:.0f}% of mass)  new-class share {new_share*100:.0f}%  r_attractor {r:+.2f}"
              f"  ({time.time()-t0:.0f}s)")

    def ms(v): return f"{st.mean(v)*100:.1f}" + (f" +/- {st.pstdev(v)*100:.1f}" if len(v) > 1 else "")
    print(f"\nforgetting severity: MNIST dropped {ms(drops)} pp")
    print(f"Q1 CONCENTRATION  top-3 destination classes swallow {ms(concs)}% of forgotten mass "
          f"(even spread over 19 would be ~16%)")
    print(f"Q2 NEW vs OLD     {ms(news)}% of forgotten mass goes to the NEW (Fashion) classes")
    print(f"Q3 SAME ATTRACTORS?  corr(destination, pre-forgetting attractor strength) = "
          f"{st.mean(r_att):+.2f}" + (f" +/- {st.pstdev(r_att):.2f}" if len(r_att) > 1 else ""))
    r = st.mean(r_att)
    print("\nverdict: " + ("forgetting REUSES the confusion attractors -> one shared mechanism"
                          if r > 0.4 else
                          "forgetting goes somewhere ELSE (mostly the new classes) -> distinct mechanisms"
                          if r < 0.2 else "weak/partial overlap — inconclusive"))


if __name__ == "__main__":
    main()
