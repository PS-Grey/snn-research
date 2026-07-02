"""Surrogate-width schedule vs fixed surrogate — does annealing the atan slope help?

The 2024-25 surrogate-gradient literature (Gygax & Zenke escape-noise theory; MSG;
adaptive/lightweight surrogates) targets gradient mismatch: a wide surrogate early in
training keeps gradients flowing through sub-threshold neurons, a sharp one late reduces
the mismatch between surrogate and true spike derivative. This tests the cheapest version:
a linear per-epoch anneal of the atan slope alpha, against fixed-alpha controls.

Conditions (Fashion-MNIST, T=10, 10 epochs, seed 0 — comparable to the RESULTS.md baseline):
  fixed2     alpha = 2 throughout (snnTorch default; = existing baseline regime)
  fixed16    alpha = 16 throughout (controls for "sharp is just better")
  anneal     alpha = 2 -> 16, linear per epoch

Usage:
    .venv/bin/python experiments/surrogate_schedule.py --epochs 10 --T 10
"""

import argparse
import time

import torch
from torch.utils.data import DataLoader

import snntorch.surrogate as surrogate

from baseline_gradient_snn import ConvSNN, get_data, n_params, pick_device, run_epoch


def set_alpha(model: ConvSNN, alpha: float) -> None:
    """Point every LIF at a fresh atan surrogate with the given slope."""
    sg = surrogate.atan(alpha=alpha)
    for lif in (model.lif1, model.lif2, model.lif_out):
        lif.spike_grad = sg


def alpha_for_epoch(schedule: str, ep: int, epochs: int) -> float:
    if schedule == "fixed2":
        return 2.0
    if schedule == "fixed16":
        return 16.0
    if schedule == "anneal":
        if epochs == 1:
            return 2.0
        return 2.0 + (16.0 - 2.0) * (ep - 1) / (epochs - 1)
    raise ValueError(schedule)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["mnist", "fashion", "cifar10"], default="fashion")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--T", type=int, default=10)
    ap.add_argument("--beta", type=float, default=0.9)
    args = ap.parse_args()

    device = pick_device()
    print(f"device: {device} | dataset: {args.dataset} | epochs: {args.epochs} | T: {args.T}")

    tr, te, in_ch, n_classes, img = get_data(args.dataset, args.batch)
    tr_loader = DataLoader(tr, batch_size=args.batch, shuffle=True, num_workers=0)
    te_loader = DataLoader(te, batch_size=256, shuffle=False, num_workers=0)

    results = {}
    for schedule in ["fixed2", "fixed16", "anneal"]:
        torch.manual_seed(0)  # identical init + batch order per condition
        model = ConvSNN(in_ch, n_classes, img, beta=args.beta, T=args.T).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=args.lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
        print(f"\n=== {schedule} ({n_params(model):,} params) ===")
        best = 0.0
        for ep in range(1, args.epochs + 1):
            alpha = alpha_for_epoch(schedule, ep, args.epochs)
            set_alpha(model, alpha)
            t0 = time.time()
            tr_acc, _ = run_epoch(model, tr_loader, device, opt)
            te_acc, _ = run_epoch(model, te_loader, device)
            sched.step()
            best = max(best, te_acc)
            print(f"  ep {ep:2d}  alpha {alpha:5.2f}  train {tr_acc*100:5.2f}%"
                  f"  test {te_acc*100:5.2f}%  ({time.time()-t0:4.1f}s)"
                  f"  spikes/img {model.last_spikes_per_img:.0f}")
        results[schedule] = best
        print(f"  best {schedule}: {best*100:.2f}%")

    print("\n" + "=" * 48)
    for schedule, best in results.items():
        print(f"  {schedule:8s} best test: {best*100:.2f}%")
    print("=" * 48)


if __name__ == "__main__":
    main()
