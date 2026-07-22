"""Heterogeneous-forgetting regime — the fair test of self-scheduled recall (Exp 19).

Exp 18 was a null: on plain MNIST every class forgets at the same rate, so *which* class the
replay budget targets can't matter. Real forgetting is heterogeneous — some things fade fast, some
stick. This builds that regime and re-runs the scheduler where it can actually bite.

Regime: MNIST + Fashion-MNIST as ONE 20-class problem (0-9 = MNIST/easy, 10-19 = Fashion/hard),
class-incremental over 10 tasks of 2 classes, MNIST first then Fashion. So difficulty varies across
the sequence and forgetting should be non-uniform.

Two questions:
  Q_forget  — is forgetting now heterogeneous, and do EASY-to-learn classes forget faster or
              slower than HARD-to-learn ones? (per-class learnability vs per-class forgetting)
  Q_sched   — with heterogeneous forgetting, does self-scheduled replay beat uniform at matched
              budget? Four arms, same replay budget/step, differ only in WHICH classes get it:
                none       — no replay (shows the raw forgetting pattern)
                uniform    — replay old classes uniformly (standard baseline)
                forgetting — replay in proportion to measured forgetting (Exp 18's crude signal)
                energy     — replay in proportion to shallow BASIN DEPTH: EP's own settled output
                             confidence on the buffer exemplar (direction 2 — the signal is the
                             SAME quantity EP uses to settle/generate, not a bolted-on accuracy)

Usage: .venv/bin/python experiments/continual_hetero.py --epochs 6 --budget 16 --buffer-k 20
"""

import argparse
import time

import torch
import torch.nn.functional as F
from torchvision import datasets, transforms

from equilibrium_prop import EPNet, pick_device, rho

NC = 20
# task orderings. "domain": MNIST (easy) first, then Fashion (hard) -- confounds difficulty w/ recency.
# "interleave": each task pairs one MNIST + its Fashion counterpart -> difficulty varies WITHIN a
# recency cohort, so easy-vs-hard forgetting can be read cleanly (class c vs class c+10, same age).
ORDERS = {
    "domain": [[2 * i, 2 * i + 1] for i in range(NC // 2)],
    "interleave": [[i, i + 10] for i in range(NC // 2)],
}
TASKS = ORDERS["domain"]


def load_2020(per_class, dev):
    tf = transforms.ToTensor()
    m_tr = datasets.MNIST("./data", train=True, download=True, transform=tf)
    m_te = datasets.MNIST("./data", train=False, download=True, transform=tf)
    f_tr = datasets.FashionMNIST("./data", train=True, download=True, transform=tf)
    f_te = datasets.FashionMNIST("./data", train=False, download=True, transform=tf)
    Xtr = torch.cat([m_tr.data, f_tr.data]).float().view(-1, 784).to(dev) / 255.0
    ytr = torch.cat([m_tr.targets, f_tr.targets + 10]).to(dev)
    Xte = torch.cat([m_te.data, f_te.data]).float().view(-1, 784).to(dev) / 255.0
    yte = torch.cat([m_te.targets, f_te.targets + 10]).to(dev)
    if per_class:                                         # cap per class to keep it snappy
        keep = torch.cat([(ytr == c).nonzero(as_tuple=True)[0][:per_class] for c in range(NC)])
        Xtr, ytr = Xtr[keep], ytr[keep]
    return Xtr, ytr, Xte, yte


@torch.no_grad()
def per_class_acc(net, Xte, yte, classes, steps):
    acc = {}
    for c in classes:
        m = yte == c
        acc[c] = (net.predict(Xte[m], steps) == c).float().mean().item()
    return acc


@torch.no_grad()
def class_energy(net, bufX, bufY, classes, steps):
    """Direction-2 signal: mean settled free-phase ENERGY per class. High energy = shallow basin
    = forgetting risk (leading indicator). This is the real EP energy, not readout confidence."""
    sig = {}
    for c in classes:
        m = bufY == c
        if m.any():
            sig[c] = net.energy(bufX[m], steps).mean().item()
    return sig


def replay_weights(schedule, net, bufX, bufY, seen, steps, dev):
    if not seen or schedule in ("none", "uniform"):
        return torch.ones(len(bufY), device=dev)
    if schedule == "forgetting":                             # retention on buffer: high = safe
        s = per_class_acc(net, bufX, bufY, seen, steps)
        return torch.tensor([(1.0 - s.get(int(bufY[i]), 1.0)) + 0.05 for i in range(len(bufY))],
                            device=dev)
    # energy: per-class risk, min-max normalised across seen classes (high energy -> more replay)
    e = class_energy(net, bufX, bufY, seen, steps)
    lo, hi = min(e.values()), max(e.values())
    rng = (hi - lo) or 1.0
    risk = {c: (e[c] - lo) / rng for c in e}                 # 0 = deepest basin, 1 = shallowest
    return torch.tensor([risk.get(int(bufY[i]), 0.5) + 0.05 for i in range(len(bufY))], device=dev)


def run(schedule, budget, buf_k, Xtr, ytr, Xte, yte, hidden, epochs, batch, beta, steps, dev):
    net = EPNet(hidden, dev, n_classes=NC)
    bufX = torch.empty(0, 784, device=dev); bufY = torch.empty(0, dtype=torch.long, device=dev)
    learnability = {}
    for cls in TASKS:
        m = torch.zeros_like(ytr, dtype=torch.bool)
        for c in cls:
            m |= (ytr == c)
        Xt, yt = Xtr[m], ytr[m]
        seen = torch.unique(bufY).tolist()
        for _ in range(epochs):
            w = replay_weights(schedule, net, bufX, bufY, seen, steps, dev)
            perm = torch.randperm(len(Xt), device=dev)
            for b in range(0, len(Xt), batch):
                idx = perm[b:b+batch]
                x, y = Xt[idx], yt[idx]
                if schedule != "none" and len(bufX):
                    r = torch.multinomial(w, min(budget, len(bufX)), replacement=len(bufX) < budget)
                    x = torch.cat([x, bufX[r]]); y = torch.cat([y, bufY[r]])
                net.train_step(x, F.one_hot(y, NC).float(), beta, steps)
        for c in cls:                                        # learnability = acc right after learning
            learnability.update(per_class_acc(net, Xte, yte, [c], steps))
            ci = (yt == c).nonzero(as_tuple=True)[0][:buf_k]
            bufX = torch.cat([bufX, Xt[ci]]); bufY = torch.cat([bufY, yt[ci]])
    final = per_class_acc(net, Xte, yte, list(range(NC)), steps)
    return learnability, final


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--budget", type=int, default=16)
    ap.add_argument("--buffer-k", type=int, default=20)
    ap.add_argument("--hidden", type=int, default=500)
    ap.add_argument("--steps", type=int, default=25)
    ap.add_argument("--beta", type=float, default=0.5)
    ap.add_argument("--per-class", type=int, default=3000)
    ap.add_argument("--order", choices=list(ORDERS), default="domain")
    args = ap.parse_args()

    global TASKS
    TASKS = ORDERS[args.order]
    dev = pick_device(); torch.manual_seed(0)
    print(f"device: {dev} | 20-class MNIST+Fashion, class-incremental | "
          f"budget {args.budget}/step, buffer {args.buffer_k}/class")
    Xtr, ytr, Xte, yte = load_2020(args.per_class, dev)

    results = {}
    for schedule in ("none", "uniform", "forgetting", "energy"):
        t0 = time.time()
        learn, final = run(schedule, args.budget, args.buffer_k, Xtr, ytr, Xte, yte,
                           args.hidden, args.epochs, args.batch, args.beta, args.steps, dev)
        results[schedule] = (learn, final)
        allc = sum(final.values()) / NC
        print(f"\n[{schedule:10s}] all-class {allc*100:.1f}%   ({time.time()-t0:.0f}s)")
        print("   class:  " + " ".join(f"{c:4d}" for c in range(NC)))
        print("   final:  " + " ".join(f"{final[c]*100:4.0f}" for c in range(NC)))
        forg = {c: learn[c] - final[c] for c in range(NC)}
        print("   forgot: " + " ".join(f"{forg[c]*100:4.0f}" for c in range(NC)))

    # Q_forget: easy-vs-hard forgetting, RECENCY-CONTROLLED, from the uniform arm (partial
    # retention so differences show; no-replay collapses to zero and is uninformative). Only clean
    # under interleave: class c (MNIST) and c+10 (Fashion) are learned in the same task -> same age.
    import statistics as st
    learn, final = results["uniform"]
    forg = {c: learn[c] - final[c] for c in range(NC)}
    print(f"\nQ_forget (uniform arm; forgetting spread std = "
          f"{st.pstdev([forg[c] for c in range(NC)])*100:.1f}pp)")
    if args.order == "interleave":
        pairs = list(range(NC // 2 - 1))                     # drop the last task (too recent)
        easy = st.mean(forg[c] for c in pairs)               # MNIST class c
        hard = st.mean(forg[c + 10] for c in pairs)          # Fashion counterpart, same recency
        print(f"   RECENCY-MATCHED (tasks 0..{NC//2-2}):  easy/MNIST forgot {easy*100:.0f}pp  "
              f"vs  hard/Fashion forgot {hard*100:.0f}pp  "
              f"-> {'EASIER-learned forget MORE' if easy>hard+.03 else 'HARDER-learned forget MORE' if hard>easy+.03 else 'about the same'}")
    else:
        print("   (order=domain confounds difficulty with recency -- run --order interleave for the "
              "clean easy-vs-hard read)")

    print("\nQ_sched (all-class acc at matched budget):")
    for s in ("uniform", "forgetting", "energy"):
        print(f"   {s:10s} {sum(results[s][1].values())/NC*100:.1f}%")


if __name__ == "__main__":
    main()
