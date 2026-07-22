"""Learned graded payload under a LOCAL rule — the honest test (Exp 22).

Exp 20 (end-to-end surrogate grad) found a learned payload adds nothing: the global readout W2
absorbs its role. Diagnosis: the payload can only earn capacity where NO global optimiser absorbs
it. This tests exactly that regime -- a fixed random reservoir (hidden NOT trained end-to-end) with
LOCAL learning only:

    hidden  : fixed random reservoir LIF (spikes carry the rate code; spike budget fixed across modes)
    readout : W2 trained by a LOCAL delta rule (output error x presynaptic message)
    payload : p_j = softplus(a_j * rate_j + b_j) -- a per-neuron INPUT-DEPENDENT graded value
              (nonlinear in the neuron's own rate, so NOT absorbable into linear W2), with a_j,b_j
              trained by a LOCAL three-factor rule: broadcast error via fixed random feedback (DFA).
              This is a plastic Delta-payload rule -- the scout's open item (a).

Message per neuron (all at MATCHED spikes -- same reservoir spikes c_j):
    binary   -- c_j                      (rate code baseline)
    computed -- sum_t s_jt * u_jt         (membrane-weighted = sigma-delta; the known method)
    fixed    -- c_j * softplus(a_j*rate+b_j), a,b RANDOM FIXED   (control: nonlinearity, not learned)
    learned  -- c_j * softplus(a_j*rate+b_j), a,b LEARNED by DFA (the Delta-payload rule)

Decisive control: learned vs fixed. If learned > fixed, LEARNING the payload (not just having a
graded nonlinearity) earns capacity under a local rule -- the novel positive Exp 20 couldn't show.
Multi-seed from the start (Exp 21 lesson: single-seed pp-comparisons are noise).

Usage: .venv/bin/python experiments/graded_payload_local.py --epochs 15 --seeds 3
"""

import argparse
import time

import torch
import torch.nn.functional as F
from torchvision import datasets, transforms


def pick_device():
    return (torch.device("mps") if torch.backends.mps.is_available()
            else torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))


def get_mnist(n_train, n_test, dev):
    tf = transforms.ToTensor()
    tr = datasets.MNIST("./data", train=True, download=True, transform=tf)
    te = datasets.MNIST("./data", train=False, download=True, transform=tf)
    Xtr = tr.data[:n_train].float().view(-1, 784).to(dev) / 255.0; ytr = tr.targets[:n_train].to(dev)
    Xte = te.data[:n_test].float().view(-1, 784).to(dev) / 255.0; yte = te.targets[:n_test].to(dev)
    return Xtr, ytr, Xte, yte


@torch.no_grad()
def reservoir(X, Wr, T, beta, thr, batch=2000):
    """Fixed random LIF reservoir. Returns per-neuron spike count c and membrane-weighted sum gmem.
    Reservoir is fixed, so this is computed ONCE per seed (features don't change during training)."""
    C, G = [], []
    for b in range(0, len(X), batch):
        I = X[b:b+batch] @ Wr.t()
        u = torch.zeros_like(I); c = torch.zeros_like(I); g = torch.zeros_like(I)
        for _ in range(T):
            u = beta * u + I
            s = (u > thr).float()
            g = g + s * u                          # membrane at spike (sigma-delta payload)
            u = u - s * thr                        # subtract reset
            c = c + s
        C.append(c); G.append(g)
    return torch.cat(C), torch.cat(G)


def message(mode, c, g, a, b, T):
    if mode == "binary":
        return c / T
    if mode == "computed":
        return g / T
    return (c / T) * F.softplus(a * (c / T) + b)   # fixed or learned: input-dependent graded payload


def run(mode, seed, Xtr_c, Xtr_g, ytr, Xte_c, Xte_g, yte, N, epochs, batch, lr2, lrp, dev):
    gth = torch.Generator(device="cpu").manual_seed(1000 + seed)
    W2 = torch.zeros(10, N, device=dev); b2 = torch.zeros(10, device=dev)
    a = (torch.randn(N, generator=gth) * 0.5 + 0.5).to(dev)      # payload params (fixed or learned)
    b = (torch.randn(N, generator=gth) * 0.5).to(dev)
    B = (torch.randn(N, 10, generator=gth) / (10 ** 0.5)).to(dev)  # fixed random DFA feedback
    learn_pay = (mode == "learned")

    with torch.no_grad():                                        # fair scale: standardise message
        m0 = message(mode, Xtr_c[:5000], Xtr_g[:5000], a, b, 1.0)
        mu, sd = m0.mean(0), m0.std(0) + 1e-6                    # fixed normalisation (readout adapts to drift)

    for _ in range(epochs):
        perm = torch.randperm(len(ytr), device=dev)
        for bi in range(0, len(ytr), batch):
            idx = perm[bi:bi+batch]
            c, g, y = Xtr_c[idx], Xtr_g[idx], ytr[idx]
            m = (message(mode, c, g, a, b, T=1.0) - mu) / sd
            out = m @ W2.t() + b2
            p = F.softmax(out, dim=1)
            e = F.one_hot(y, 10).float() - p                     # output error (local)
            W2 += lr2 * (e.t() @ m) / len(idx)                   # delta rule: error x presynaptic
            b2 += lr2 * e.mean(0)
            if learn_pay:
                delta = e @ B.t()                                # broadcast error via random feedback (DFA)
                z = a * c + b                                     # c is the neuron's rate
                dz = delta * (c * torch.sigmoid(z) / sd)         # d loss / d z (through normalised message)
                a += lrp * (dz * c).mean(0)                       # dz/da = c
                b += lrp * dz.mean(0)                             # dz/db = 1

    with torch.no_grad():
        m = (message(mode, Xte_c, Xte_g, a, b, T=1.0) - mu) / sd
        acc = ((m @ W2.t() + b2).argmax(1) == yte).float().mean().item()
    return acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--hidden", type=int, default=800)
    ap.add_argument("--T", type=int, default=20)
    ap.add_argument("--beta", type=float, default=0.9)
    ap.add_argument("--thr", type=float, default=1.0)
    ap.add_argument("--scale", type=float, default=4.0)
    ap.add_argument("--lr2", type=float, default=0.2)
    ap.add_argument("--lrp", type=float, default=0.05)
    ap.add_argument("--n-train", type=int, default=60000)
    ap.add_argument("--n-test", type=int, default=10000)
    args = ap.parse_args()

    dev = pick_device()
    print(f"device: {dev} | graded payload under LOCAL rule | reservoir {args.hidden}, T {args.T}, "
          f"{args.seeds} seeds")
    Xtr, ytr, Xte, yte = get_mnist(args.n_train, args.n_test, dev)

    modes = ("binary", "computed", "fixed", "learned")
    accs = {m: [] for m in modes}; spk = 0.0
    for seed in range(args.seeds):
        torch.manual_seed(seed)
        gcpu = torch.Generator(device="cpu").manual_seed(seed)
        Wr = (torch.randn(args.hidden, 784, generator=gcpu) * (args.scale / 784 ** 0.5)).to(dev)
        # precompute reservoir features once; divide counts by T here so message() rate is c/T
        trc, trg = reservoir(Xtr, Wr, args.T, args.beta, args.thr)
        tec, teg = reservoir(Xte, Wr, args.T, args.beta, args.thr)
        spk = trc.sum(1).mean().item()
        trc, trg, tec, teg = trc / args.T, trg / args.T, tec / args.T, teg / args.T
        for m in modes:
            accs[m].append(run(m, seed, trc, trg, ytr, tec, teg, yte, args.hidden,
                               args.epochs, args.batch, args.lr2, args.lrp, dev))

    print(f"\nspikes/img (matched across modes): {spk:.0f}")
    print(f"{'mode':10s} {'mean acc':>9s} {'per-seed':>28s}")
    for m in modes:
        v = accs[m]
        mean = sum(v) / len(v)
        print(f"{m:10s} {mean*100:8.2f}%   [" + " ".join(f"{x*100:5.2f}" for x in v) + "]")
    lm = sum(accs["learned"]) / args.seeds; fm = sum(accs["fixed"]) / args.seeds
    cm = sum(accs["computed"]) / args.seeds; bm = sum(accs["binary"]) / args.seeds
    print(f"\nlearned - fixed  = {(lm-fm)*100:+.2f}pp  (does LEARNING the payload beat a fixed nonlin?)")
    print(f"learned - computed = {(lm-cm)*100:+.2f}pp  |  computed - binary = {(cm-bm)*100:+.2f}pp")


if __name__ == "__main__":
    main()
