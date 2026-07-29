"""Does EASY-learned forget faster? Multi-seed, ceiling-controlled (Exp 25).

Exp 19 found easier-learned classes forget MORE (MNIST 35pp vs Fashion 23pp at matched recency),
but it was SINGLE-SEED and had a ceiling confound I flagged at the time: MNIST is learned to a higher
accuracy, so it has more absolute percentage points to lose. This settles it properly:

  - 5 seeds (standing rule after the Exp 21 seed-luck retraction)
  - recency-controlled: interleaved tasks, so MNIST class c and Fashion class c+10 are learned in the
    SAME task and have aged identically by the end
  - BOTH metrics reported:
      absolute forgetting  = learned - final            (the Exp 19 metric; ceiling-confounded)
      relative retention   = final / learned            (ceiling-CONTROLLED: what FRACTION was kept)
    If easy-forgets-more survives on RELATIVE retention, the effect is real, not a ceiling artefact.

Uniform replay (partial retention, so differences are visible; no-replay collapses everything to 0).
Usage: .venv/bin/python experiments/forgetting_asymmetry.py --seeds 5
"""

import argparse
import statistics as st
import time

import torch
import torch.nn.functional as F

from equilibrium_prop import EPNet, pick_device
from continual_hetero import NC, load_2020, per_class_acc

TASKS = [[i, i + 10] for i in range(10)]      # interleave: MNIST i paired with Fashion i, same age


def run(seed, budget, buf_k, Xtr, ytr, Xte, yte, hidden, epochs, batch, beta, steps, dev):
    net = EPNet(hidden, dev, n_classes=NC, seed=seed)
    bufX = torch.empty(0, 784, device=dev); bufY = torch.empty(0, dtype=torch.long, device=dev)
    learn = {}
    for cls in TASKS:
        m = torch.zeros_like(ytr, dtype=torch.bool)
        for c in cls:
            m |= (ytr == c)
        Xt, yt = Xtr[m], ytr[m]
        for _ in range(epochs):
            perm = torch.randperm(len(Xt), device=dev)
            for b in range(0, len(Xt), batch):
                idx = perm[b:b+batch]
                x, y = Xt[idx], yt[idx]
                if len(bufX):                                   # uniform replay
                    r = torch.randint(0, len(bufX), (min(budget, len(bufX)),), device=dev)
                    x = torch.cat([x, bufX[r]]); y = torch.cat([y, bufY[r]])
                net.train_step(x, F.one_hot(y, NC).float(), beta, steps)
        for c in cls:                                            # learnability: acc right after learning
            learn.update(per_class_acc(net, Xte, yte, [c], steps))
            ci = (yt == c).nonzero(as_tuple=True)[0][:buf_k]
            bufX = torch.cat([bufX, Xt[ci]]); bufY = torch.cat([bufY, yt[ci]])
    final = per_class_acc(net, Xte, yte, list(range(NC)), steps)
    return learn, final


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--budget", type=int, default=16)
    ap.add_argument("--buffer-k", type=int, default=20)
    ap.add_argument("--hidden", type=int, default=500)
    ap.add_argument("--steps", type=int, default=25)
    ap.add_argument("--beta", type=float, default=0.5)
    ap.add_argument("--per-class", type=int, default=3000)
    args = ap.parse_args()

    dev = pick_device(); torch.manual_seed(0)
    print(f"device: {dev} | easy-vs-hard forgetting | interleaved 20-class, {args.seeds} seeds")
    Xtr, ytr, Xte, yte = load_2020(args.per_class, dev)

    abs_e, abs_h, rel_e, rel_h = [], [], [], []
    dropped, by_task = [], {}                                    # unlearned pairs; retention vs task age
    for seed in range(args.seeds):
        t0 = time.time()
        learn, final = run(seed, args.budget, args.buffer_k, Xtr, ytr, Xte, yte, args.hidden,
                           args.epochs, args.batch, args.beta, args.steps, dev)
        # MATCHED POPULATION: a class counts only if BOTH members of its pair actually learned.
        # (Earlier version excluded never-learned classes from the relative metric but not the
        # absolute one, biasing absolute hard-class forgetting downward — classes that never
        # learned were recorded as forgetting nothing.)
        pairs = [c for c in range(9) if learn[c] > 0.05 and learn[c+10] > 0.05]
        dropped.append(9 - len(pairs))
        if not pairs:
            continue
        ae = st.mean(learn[c] - final[c] for c in pairs)
        ah = st.mean(learn[c+10] - final[c+10] for c in pairs)
        re = st.mean(final[c] / learn[c] for c in pairs)
        rh = st.mean(final[c+10] / learn[c+10] for c in pairs)
        abs_e.append(ae); abs_h.append(ah); rel_e.append(re); rel_h.append(rh)
        # recency: is the retained fraction constant, or does it decay with task age?
        for c in pairs:
            by_task.setdefault(c, []).append(st.mean([final[c]/learn[c], final[c+10]/learn[c+10]]))
        print(f"  seed {seed}: abs easy {ae*100:5.1f}pp hard {ah*100:5.1f}pp | "
              f"rel-retention easy {re*100:5.1f}% hard {rh*100:5.1f}%  "
              f"(dropped {dropped[-1]} unlearned pairs, {time.time()-t0:.0f}s)")

    def ms(v): return f"{st.mean(v)*100:5.1f} +/- {(st.pstdev(v)*100 if len(v)>1 else 0):.1f}"
    print(f"\n(matched population: dropped {st.mean(dropped):.1f}/9 pairs per seed where a class never learned)")
    print(f"\nRECENCY — retained fraction by task age (task 0 = oldest, most erosion):")
    print("   task:      " + " ".join(f"{c:5d}" for c in sorted(by_task)))
    print("   retained:  " + " ".join(f"{st.mean(by_task[c])*100:4.0f}%" for c in sorted(by_task)))
    print(f"\nABSOLUTE forgetting (pp, ceiling-CONFOUNDED):")
    print(f"   easy/MNIST   {ms(abs_e)}      hard/Fashion {ms(abs_h)}")
    print(f"RELATIVE retention (% of learned kept, ceiling-CONTROLLED — the honest metric):")
    print(f"   easy/MNIST   {ms(rel_e)}      hard/Fashion {ms(rel_h)}")
    de = st.mean(rel_h) - st.mean(rel_e)                     # >0 means HARD kept less => hard forgets more
    if args.seeds < 2:
        print("\nverdict: single seed — cannot judge (rerun with --seeds 5)")
        return
    spread = max(st.pstdev(rel_e), st.pstdev(rel_h))
    verdict = ("EASY forgets more (Exp 19 SURVIVES ceiling control)" if -de > 2 * spread and -de > 0.02 else
               "HARD forgets more (Exp 19 REVERSES once the ceiling is controlled)" if de > 2 * spread and de > 0.02 else
               "NO robust difference — Exp 19's effect was the ceiling artefact")
    print(f"\nverdict: {verdict}  (relative gap {de*100:+.1f}pp hard-minus-easy, seed spread {spread*100:.1f}pp)")


if __name__ == "__main__":
    main()
