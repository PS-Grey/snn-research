"""Timestep (T) sweep — the accuracy vs energy curve for the gradient-SNN.

A closed SNN<->ANN gap is only useful for neuromorphic deployment if it survives at low T:
fewer timesteps means fewer spikes (lower SynOps / energy) and lower latency on Loihi/Speck.
This sweeps T over the SNN from baseline_gradient_snn and records test accuracy and the
mean spikes/image (the SynOps proxy) at each T, on Fashion-MNIST.

Usage:
    .venv/bin/python experiments/timestep_sweep.py --epochs 10
"""

import argparse
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from baseline_gradient_snn import ConvSNN, get_data, pick_device, run_epoch, n_params


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["mnist", "fashion", "cifar10"], default="fashion")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--beta", type=float, default=0.9)
    ap.add_argument("--Ts", type=int, nargs="+", default=[1, 2, 4, 8])
    args = ap.parse_args()

    device = pick_device()
    print(f"device: {device} | dataset: {args.dataset} | epochs/T: {args.epochs} | Ts: {args.Ts}")

    tr, te, in_ch, n_classes, img = get_data(args.dataset, args.batch)
    tr_loader = DataLoader(tr, batch_size=args.batch, shuffle=True, num_workers=0)
    te_loader = DataLoader(te, batch_size=256, shuffle=False, num_workers=0)

    rows = []
    for T in args.Ts:
        torch.manual_seed(0)
        model = ConvSNN(in_ch, n_classes, img, beta=args.beta, T=T).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=args.lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
        print(f"\n=== T={T}  ({n_params(model):,} params) ===")
        best, best_spikes = 0.0, 0.0
        for ep in range(1, args.epochs + 1):
            t0 = time.time()
            tr_acc, _ = run_epoch(model, tr_loader, device, opt)
            te_acc, _ = run_epoch(model, te_loader, device)
            sched.step()
            if te_acc > best:
                best, best_spikes = te_acc, float(model.last_spikes_per_img)
            print(f"  ep {ep:2d}  train {tr_acc*100:5.2f}%  test {te_acc*100:5.2f}%"
                  f"  ({time.time()-t0:5.1f}s) | spikes/img {model.last_spikes_per_img:.0f}")
        rows.append((T, best, best_spikes))
        print(f"  T={T}: best test {best*100:.2f}%  @ ~{best_spikes:.0f} spikes/img")

    print("\n" + "=" * 52)
    print(f"  {'T':>3}  {'test acc':>9}  {'spikes/img':>11}")
    for T, acc, sp in rows:
        print(f"  {T:>3}  {acc*100:8.2f}%  {sp:>11.0f}")
    print("=" * 52)


if __name__ == "__main__":
    main()
