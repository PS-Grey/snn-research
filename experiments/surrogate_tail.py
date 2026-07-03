"""Surrogate tail shape vs width — is the load-bearing variable the tail, not the width?

Prior sigmoid scale sweeps (vault, 30 May) found surrogate sharpness load-bearing: the conv-SNN
collapsed from ~92% to ~81% as the sigmoid was sharpened (scale 5 -> 25). But the atan
surrogate-width test (surrogate_schedule.py) showed NO collapse across α 2->16, even though
atan α=16 is by half-width *sharper* than the sigmoid scale=25 that collapsed. Hypothesis:
the difference is the surrogate's TAIL, not its peak width. Sigmoid's gradient decays
exponentially away from threshold, so sharpening starves off-threshold neurons; atan's is
Lorentzian (1/(1+x²)) with heavy polynomial tails that keep a small gradient flowing.

Design: match the two families at equal gradient HALF-WIDTH (peak sharpness), then sharpen
both. Only the tail differs. Prediction: at sharp half-width, sigmoid collapses, atan holds.

Same conv-BN-LIF backbone as the baseline (ConvSNN), Fashion-MNIST, T=10, seed 0.

Usage:
    .venv/bin/python experiments/surrogate_tail.py --epochs 6 --T 10
"""

import argparse
import time

import torch
from torch.utils.data import DataLoader

import snntorch.surrogate as surrogate

from baseline_gradient_snn import ConvSNN, get_data, n_params, pick_device, run_epoch


def make_stable_sigmoid(slope: float):
    """Logistic-sigmoid surrogate via torch.sigmoid (overflow-safe).

    snnTorch's surrogate.sigmoid computes its backward with a bare exp(), which overflows to
    inf -> NaN for neurons more than ~20 below threshold — routine in a BN conv-SNN — silently
    killing training. torch.sigmoid is numerically stable. Exponential tails are preserved (the
    property under test); only the overflow is removed. Matches the 30-May hand-rolled surrogate.
    """
    class StableSigmoid(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            ctx.save_for_backward(x)
            ctx.slope = slope
            return (x > 0).float()

        @staticmethod
        def backward(ctx, grad):
            (x,) = ctx.saved_tensors
            sig = torch.sigmoid(ctx.slope * x)
            return grad * ctx.slope * sig * (1 - sig)

    return StableSigmoid.apply


def make_surrogate(family: str, param: float):
    if family == "atan":
        return surrogate.atan(alpha=param)
    if family == "sigmoid":
        return make_stable_sigmoid(param)
    raise ValueError(family)


def grad_at(fn, xs: torch.Tensor) -> torch.Tensor:
    """Surrogate gradient magnitude at membrane offsets xs (measured through autograd)."""
    x = xs.clone().detach().requires_grad_(True)
    fn(x).sum().backward()
    return x.grad.abs()


def half_width(fn) -> float:
    """Membrane offset where the surrogate gradient falls to half its peak (at x=0)."""
    peak = grad_at(fn, torch.tensor([0.0])).item()
    lo, hi = 0.0, 20.0
    for _ in range(60):
        mid = (lo + hi) / 2
        g = grad_at(fn, torch.tensor([mid])).item()
        if g > 0.5 * peak:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def param_for_halfwidth(family: str, target_hw: float) -> float:
    """Bisection: find the family's sharpness parameter that yields target half-width."""
    lo, hi = 1e-3, 5000.0  # larger param => sharper => smaller half-width
    for _ in range(80):
        mid = (lo + hi) / 2
        hw = half_width(make_surrogate(family, mid))
        if hw > target_hw:      # too smooth, sharpen
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def tail_ratio(fn, at: float = 0.3) -> float:
    """Gradient at |V|=at relative to peak — how much signal reaches off-threshold neurons."""
    peak = grad_at(fn, torch.tensor([0.0])).item()
    return grad_at(fn, torch.tensor([at])).item() / peak


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="fashion")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--T", type=int, default=10)
    ap.add_argument("--beta", type=float, default=0.9)
    args = ap.parse_args()

    device = pick_device()
    print(f"device: {device} | dataset: {args.dataset} | epochs: {args.epochs} | T: {args.T}")

    # Half-widths spanning the smooth plateau -> the sharp regime where the old sigmoid collapsed.
    targets = [0.35, 0.10, 0.05, 0.02]

    # Resolve each family's parameter to hit each half-width, and characterise the tail.
    print("\nmatched sharpness (half-width -> param, tail@0.3):")
    plan = []
    for hw in targets:
        for fam in ("atan", "sigmoid"):
            p = param_for_halfwidth(fam, hw)
            tr = tail_ratio(make_surrogate(fam, p))
            plan.append((hw, fam, p, tr))
            print(f"  hw {hw:.3f}  {fam:8s} param {p:8.3f}  tail@0.3 {tr:.4f}")

    tr_data, te_data, in_ch, n_classes, img = get_data(args.dataset, args.batch)
    tr_loader = DataLoader(tr_data, batch_size=args.batch, shuffle=True, num_workers=0)
    te_loader = DataLoader(te_data, batch_size=256, shuffle=False, num_workers=0)

    results = []
    for hw, fam, param, tr in plan:
        torch.manual_seed(0)  # identical init + batch order across every condition
        model = ConvSNN(in_ch, n_classes, img, beta=args.beta, T=args.T).to(device)
        sg = make_surrogate(fam, param)
        for lif in (model.lif1, model.lif2, model.lif_out):
            lif.spike_grad = sg
        opt = torch.optim.Adam(model.parameters(), lr=args.lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
        print(f"\n=== {fam} hw={hw:.3f} (param {param:.3f}, tail@0.3 {tr:.4f}) ===")
        best = 0.0
        for ep in range(1, args.epochs + 1):
            t0 = time.time()
            tr_acc, _ = run_epoch(model, tr_loader, device, opt)
            te_acc, _ = run_epoch(model, te_loader, device)
            sched.step()
            best = max(best, te_acc)
            print(f"  ep {ep:2d}  train {tr_acc*100:5.2f}%  test {te_acc*100:5.2f}%"
                  f"  ({time.time()-t0:4.1f}s)")
        results.append((hw, fam, tr, best))
        print(f"  best: {best*100:.2f}%")

    print("\n" + "=" * 60)
    print(f"  {'half-width':>10} | {'atan best':>10} | {'sigmoid best':>12} | {'atan tail':>9} | {'sig tail':>8}")
    print("  " + "-" * 58)
    for hw in targets:
        a = next((b for h, f, t, b in results if h == hw and f == "atan"), None)
        s = next((b for h, f, t, b in results if h == hw and f == "sigmoid"), None)
        at = next((t for h, f, t, b in results if h == hw and f == "atan"), None)
        st = next((t for h, f, t, b in results if h == hw and f == "sigmoid"), None)
        print(f"  {hw:>10.3f} | {a*100:>9.2f}% | {s*100:>11.2f}% | {at:>9.4f} | {st:>8.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
