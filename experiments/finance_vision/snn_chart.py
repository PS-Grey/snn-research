"""SNN chart-vision: predict P(next-7d return > 0) from 64 daily OHLC bars.

From the finance project's benchmark (see the briefing). The point of using an SNN here, versus
the from-scratch CNN that found nothing: a CNN sees the chart as a STATIC image. An SNN reads it
as a TIME SERIES -- the 64 bars become the 64 timesteps of the network, and membrane potential
carries memory forward bar by bar. That temporal inductive bias is the hypothesis under test.

This is the surrogate-gradient (backprop) baseline: fastest to an honest arena number and the
fair head-to-head with the CNN. The online / STDP self-correcting version is the follow-up.

Task (fixed, do not redefine): input 64x4 price-normalised OHLC; output P(fwd7 > 0). Train only
on split=="train" (dates < 2025-01-01). Temporal split is sacred -- no peeking at test labels.

Output: writes preds_snn.csv (id,p) for every test row into the finance vision_ds dir, then
score with finance/lab/arena_score.py.

Usage:
    .venv/bin/python experiments/finance_vision/snn_chart.py --epochs 30
"""

import argparse
import csv
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import snntorch as snn
from snntorch import surrogate

DS = os.path.expanduser("~/Claude/finance/lab/data/vision_ds")


def pick_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def bar_features(ohlc):
    """OHLC [N,64,4] -> per-bar event/temporal features [N,64,F]. Deltas & bar shape, not raw
    levels: an event stream cares about change, not absolute price."""
    o, h, l, c = ohlc[..., 0], ohlc[..., 1], ohlc[..., 2], ohlc[..., 3]
    eps = 1e-6
    prev_c = np.concatenate([c[:, :1], c[:, :-1]], axis=1)
    ret = (c - prev_c) / (prev_c + eps)          # bar-to-bar return (the core 'event')
    rng = (h - l) / (c + eps)                     # volatility of the bar
    body = (c - o) / (c + eps)                    # bar body (green/red magnitude)
    uwick = (h - np.maximum(o, c)) / (c + eps)    # upper wick
    lwick = (np.minimum(o, c) - l) / (c + eps)    # lower wick
    feats = np.stack([ret, rng, body, uwick, lwick], axis=-1).astype(np.float32)
    return feats                                  # [N,64,5]


def load():
    ohlc = np.load(os.path.join(DS, "raw.npz"))["ohlc"]
    rows = list(csv.DictReader(open(os.path.join(DS, "meta.csv"))))
    ids = np.array([r["id"] for r in rows])
    y = np.array([int(r["label"]) for r in rows], dtype=np.float32)
    split = np.array([r["split"] for r in rows])
    X = bar_features(ohlc)
    tr, te = split == "train", split == "test"
    # standardise features on TRAIN only (no leakage)
    mu = X[tr].reshape(-1, X.shape[-1]).mean(0)
    sd = X[tr].reshape(-1, X.shape[-1]).std(0) + 1e-6
    X = (X - mu) / sd
    return (X[tr], y[tr]), (X[te], y[te], ids[te])


class ChartSNN(nn.Module):
    """Reads the 64 bars as 64 timesteps. LIF membranes carry temporal memory across bars;
    the output membrane accumulates evidence, read out at the end as a single up/down logit."""

    def __init__(self, n_feat=5, hidden=64, beta=0.9):
        super().__init__()
        sg = surrogate.atan()
        lif = lambda: snn.Leaky(beta=beta, spike_grad=sg, reset_mechanism="subtract")
        self.fc1 = nn.Linear(n_feat, hidden)
        self.bn1 = nn.BatchNorm1d(hidden)
        self.lif1 = lif()
        self.fc2 = nn.Linear(hidden, hidden)
        self.bn2 = nn.BatchNorm1d(hidden)
        self.lif2 = lif()
        self.fc_out = nn.Linear(hidden, 1)
        self.lif_out = snn.Leaky(beta=beta, spike_grad=sg, reset_mechanism="none", output=True)

    def forward(self, x):                         # x: [B, 64, F]
        m1 = self.lif1.init_leaky(); m2 = self.lif2.init_leaky(); mo = self.lif_out.init_leaky()
        mem_sum = 0.0
        spk = 0.0
        for t in range(x.size(1)):                # walk the 64 bars in order
            c1 = self.bn1(self.fc1(x[:, t]))
            s1, m1 = self.lif1(c1, m1)
            c2 = self.bn2(self.fc2(s1))
            s2, m2 = self.lif2(c2, m2)
            _, mo = self.lif_out(self.fc_out(s2), mo)
            mem_sum = mem_sum + mo
            spk = spk + s1.sum() + s2.sum()
        self.last_spikes = spk / x.size(0)
        return mem_sum.squeeze(-1) / x.size(1)    # accumulated up/down evidence -> logit


def run_epoch(model, loader, device, opt=None):
    train = opt is not None
    model.train(train)
    tot, correct, loss_sum = 0, 0, 0.0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        with torch.set_grad_enabled(train):
            logit = model(xb)
            loss = F.binary_cross_entropy_with_logits(logit, yb)
            if train:
                opt.zero_grad(); loss.backward(); opt.step()
        loss_sum += loss.item() * xb.size(0)
        correct += ((logit > 0).float() == yb).sum().item()
        tot += xb.size(0)
    return correct / tot, loss_sum / tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="event-stream 64-bar LIF, surrogate-grad")
    args = ap.parse_args()

    device = pick_device(); torch.manual_seed(args.seed)
    (Xtr, ytr), (Xte, yte, ids_te) = load()
    print(f"device: {device} | train {len(Xtr)} test {len(Xte)} | up-rate tr {ytr.mean():.3f}")

    tr_loader = DataLoader(TensorDataset(torch.tensor(Xtr), torch.tensor(ytr)),
                           batch_size=args.batch, shuffle=True, num_workers=0)
    te_loader = DataLoader(TensorDataset(torch.tensor(Xte), torch.tensor(yte)),
                           batch_size=256, shuffle=False, num_workers=0)

    model = ChartSNN(n_feat=Xtr.shape[-1], hidden=args.hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    print(f"params: {sum(p.numel() for p in model.parameters()):,}")

    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        tr_acc, tr_loss = run_epoch(model, tr_loader, device, opt)
        te_acc, _ = run_epoch(model, te_loader, device)
        sched.step()
        if ep % 5 == 0 or ep == 1:
            print(f"  ep {ep:2d}  train {tr_acc*100:5.2f}%  test {te_acc*100:5.2f}%"
                  f"  loss {tr_loss:.3f}  ({time.time()-t0:.1f}s)  spk/img {model.last_spikes:.0f}")

    # predictions for the arena (probabilities, not just class)
    model.eval()
    ps = []
    with torch.no_grad():
        for xb, _ in te_loader:
            ps.append(torch.sigmoid(model(xb.to(device))).cpu())
    p = torch.cat(ps).numpy()
    out = os.path.join(DS, "preds_snn.csv")
    with open(out, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["id", "p"])
        for i, pi in zip(ids_te, p):
            w.writerow([i, f"{pi:.6f}"])
    print(f"\nwrote {len(p)} preds -> {out}")
    print(f"test acc {(( (p>0.5)==(yte>0.5) ).mean())*100:.2f}%  (base rate maj-class ~54.3%)")
    print(f"\nscore it:\n  ~/Claude/finance/lab/.venv/bin/python "
          f"~/Claude/finance/lab/arena_score.py preds_snn.csv \"{args.tag}\"")


if __name__ == "__main__":
    main()
