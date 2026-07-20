"""snn-online: walk-forward self-correcting SNN. The one edge a batch GBM structurally can't copy.

Frozen SNN feature extractor (batch-trained 2022-2024) + an online-adapting linear readout
(delta rule = our validated surprise-weighted local rule). Walk forward through 2025-2026: at
date t, predict, then learn ONLY from samples whose 7-day label resolved (date <= t-7). Never an
unresolved outcome.

Fairness: the online learning rate is NOT tuned on the test set (that would void the row). It is
selected on a TRAIN-period walk-forward holdout (last slice of 2022-2024), then applied to the
test once. Prices/flow up to t always fine; labels usable iff dated <= t-7d. Tag: snn-online.

Compares frozen readout (batch) vs online readout on the same features: does tracking the regime
shift add value, or is the flow signal stationary (online just tracks, doesn't extract more)?

Usage: .venv/bin/python experiments/finance_vision/snn_online.py
"""

import argparse
import csv
import os
from datetime import datetime, timedelta

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from snn_chart import ChartSNN, load, run_epoch, pick_device, DS


def extract_features(model, X, device, bs=256):
    """Frozen SNN -> per-sample feature = summed lif2 spikes over the 64 bars [N, hidden]."""
    model.eval()
    feats = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            xb = torch.tensor(X[i:i+bs]).to(device)
            m1 = model.lif1.init_leaky(); m2 = model.lif2.init_leaky()
            acc = 0.0
            for t in range(xb.size(1)):
                c1 = model.bn1(model.fc1(xb[:, t])); s1, m1 = model.lif1(c1, m1)
                c2 = model.bn2(model.fc2(s1)); s2, m2 = model.lif2(c2, m2)
                acc = acc + s2
            feats.append(acc.cpu())
    return torch.cat(feats).numpy()


def train_readout(F, y, epochs, lr, seed=0):
    w = np.zeros(F.shape[1], np.float32); b = 0.0
    rng = np.random.default_rng(seed)
    for _ in range(epochs):
        for i in rng.permutation(len(F)):
            p = 1 / (1 + np.exp(-(F[i] @ w + b)))
            err = y[i] - p
            w += lr * err * F[i]; b += lr * err
    return w, b.astype(np.float32) if hasattr(b, "astype") else np.float32(b)


def walk_forward(F, y, dates, w0, b0, online_lr):
    """Predict each date with a readout that has learned only from labels resolved by then
    (date <= t-7d). Returns per-sample online predictions."""
    w, b = w0.copy(), float(b0)
    p = np.zeros(len(F), np.float32)
    learned = np.zeros(len(F), bool)
    for t in sorted(set(dates)):
        for i in np.where((dates <= (t - timedelta(days=7))) & ~learned)[0]:
            err = y[i] - 1 / (1 + np.exp(-(F[i] @ w + b)))
            w += online_lr * err * F[i]; b += online_lr * err
            learned[i] = True
        for i in np.where(dates == t)[0]:
            p[i] = 1 / (1 + np.exp(-(F[i] @ w + b)))
    return p


def decile_spread(p, fwd):
    k = len(p) // 10; o = np.argsort(p)
    return fwd[o[-k:]].mean() - fwd[o[:k]].mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--readout-epochs", type=int, default=20)
    ap.add_argument("--readout-lr", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = pick_device(); torch.manual_seed(args.seed); np.random.seed(args.seed)
    (Xtr, ytr), (Xte, yte, ids_te, fwd_te) = load(flow=True, encoding="naive")
    rows = list(csv.DictReader(open(os.path.join(DS, "meta.csv"))))
    dt = {r["id"]: datetime.strptime(r["date"], "%Y-%m-%d") for r in rows}
    ids_tr = np.array([r["id"] for r in rows if r["split"] == "train"])
    fwd_tr = np.array([float(r["fwd7"]) for r in rows if r["split"] == "train"], np.float32)
    dtr = np.array([dt[i] for i in ids_tr]); dte = np.array([dt[i] for i in ids_te])
    print(f"device: {device} | train {len(Xtr)} test {len(Xte)}")

    # 1. batch-train SNN feature extractor on train, freeze
    trl = DataLoader(TensorDataset(torch.tensor(Xtr), torch.tensor(ytr)), batch_size=128, shuffle=True)
    model = ChartSNN(n_feat=Xtr.shape[-1], hidden=64).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    print("pretraining SNN features (frozen after)...")
    for _ in range(args.epochs):
        run_epoch(model, trl, device, opt); sched.step()
    Ftr = extract_features(model, Xtr, device)
    Fte = extract_features(model, Xte, device)
    mu, sd = Ftr.mean(0), Ftr.std(0) + 1e-6
    Ftr = (Ftr - mu) / sd; Fte = (Fte - mu) / sd

    # 2-3. Multi-seed the readout on the FIXED features (the readout order is the big variance
    # source). Per seed: fair online-lr selection on a train holdout (never test), then frozen
    # vs online on test. The paired (online - frozen) delta is the clean "does online help".
    cut = np.quantile([d.timestamp() for d in dtr], 0.75)
    core = np.array([d.timestamp() < cut for d in dtr]); val = ~core
    n_seeds = 8
    fr, on, dl, lrs = [], [], [], []
    p_sum = np.zeros(len(Fte), np.float32)
    print(f"\nmulti-seed ({n_seeds}) readout, fair train-holdout lr each:")
    for s in range(n_seeds):
        w_c, b_c = train_readout(Ftr[core], ytr[core], args.readout_epochs, args.readout_lr, seed=s)
        best_lr, best_v = 0.0, -1e9
        for lr in (0.0, 0.001, 0.003, 0.01, 0.03):
            sv = decile_spread(walk_forward(Ftr[val], ytr[val], dtr[val], w_c, b_c, lr), fwd_tr[val])
            if sv > best_v:
                best_v, best_lr = sv, lr
        w0, b0 = train_readout(Ftr, ytr, args.readout_epochs, args.readout_lr, seed=s)
        pf = 1 / (1 + np.exp(-(Fte @ w0 + b0)))
        po = walk_forward(Fte, yte, dte, w0, b0, best_lr)
        sf, so = decile_spread(pf, fwd_te), decile_spread(po, fwd_te)
        fr.append(sf); on.append(so); dl.append(so - sf); lrs.append(best_lr); p_sum += po
        print(f"  seed {s}: lr {best_lr:.3f}  frozen {sf*100:+.2f}%  online {so*100:+.2f}%  "
              f"delta {(so-sf)*100:+.2f}pp")

    fr, on, dl = np.array(fr), np.array(on), np.array(dl)
    with open(os.path.join(DS, "preds_snn-online.csv"), "w", newline="") as f:
        wr = csv.writer(f); wr.writerow(["id", "p"])
        for i, pi in zip(ids_te, p_sum / n_seeds):
            wr.writerow([i, f"{pi:.6f}"])

    print("\n" + "=" * 62)
    print(f"  frozen  spread: {fr.mean()*100:+.2f}% mean  (range {fr.min()*100:+.2f}..{fr.max()*100:+.2f})")
    print(f"  ONLINE  spread: {on.mean()*100:+.2f}% mean  (range {on.min()*100:+.2f}..{on.max()*100:+.2f})")
    print(f"  online - frozen: {dl.mean()*100:+.2f}pp mean  (range {dl.min()*100:+.2f}..{dl.max()*100:+.2f})")
    print(f"  chosen lrs: {lrs}")
    print(f"  (GBM +1.43%, snn-flow +1.21%, noise floor +0.47%)")
    print("=" * 62)


if __name__ == "__main__":
    main()
