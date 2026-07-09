"""Unsupervised STDP on MNIST — the backprop-free baseline (Diehl & Cook 2015 style).

The first experiment of the post-pivot direction: learn MNIST with NO backpropagation, using
only a local, on-chip-implementable plasticity rule. This is the number a local rule has to
grow from, set against the surrogate-gradient reference (98.9%).

Everything here maps to neuromorphic hardware:
- **Trace-based STDP** (not pairwise all-to-all): each synapse keeps a decaying pre-trace; on a
  post-synaptic spike the weight moves toward that trace. This is exactly the form Loihi runs
  on-chip — O(1) state per synapse, no history buffer.
- **Lateral inhibition** (winner-take-all): a spiking excitatory neuron suppresses the others,
  so different neurons specialise to different digits. On-chip = fixed inhibitory synapses.
- **Adaptive threshold** (homeostasis): a neuron that fires often raises its own threshold, so
  no neuron hogs every pattern. On-chip = per-neuron threshold state.
- **Weight normalisation**: each neuron's input weights sum to a constant, enforcing
  competition. On-chip = periodic per-neuron rescale.

No labels are used during learning. Labels are assigned to neurons afterward by their responses
(a readout step), then test images are classified by which labelled neurons fire most.

Noise knob (`--noise-std`): Gaussian membrane noise during learning, transient per-timestep,
off at eval. Membrane noise is this project's recurring theme and is on-chip-plausible (real
and neuromorphic neurons have intrinsic voltage noise). Tests whether it helps STDP too.

Usage:
    .venv/bin/python experiments/stdp_diehl_cook.py --n-exc 400 --n-train 15000 --n-test 10000
    .venv/bin/python experiments/stdp_diehl_cook.py --noise-std 0.1     # noise variant
"""

import argparse
import time

import torch
from torchvision import datasets, transforms


# --------------------------------------------------------------------------- #
# Hyperparameters (tuned so a specialised neuron fires a few times per image)  #
# --------------------------------------------------------------------------- #
V_DECAY = 0.92        # membrane leak per timestep (tau ~ 12 steps)
TR_DECAY = 0.92       # pre-synaptic trace decay
V_THRESH = 0.5        # base firing threshold (weights normalise to sum 1, so I <= 1/timestep)
V_RESET = 0.0
THETA_PLUS = 0.05     # homeostatic threshold bump per spike
THETA_DECAY = 1.0     # persistent homeostasis during training (D&C tau is ~1e7 ms ≈ constant)
ETA = 0.01            # STDP learning rate
W_MAX = 1.0
NORM_TARGET = 1.0     # each neuron's input weights sum to this (competition)
INTENSITY = 0.30      # Poisson spike prob per timestep = pixel * INTENSITY


class STDPNet:
    """Input -> excitatory layer, trace-STDP plastic synapses, WTA lateral inhibition."""

    def __init__(self, n_input, n_exc, device, seed=0, theta_plus=THETA_PLUS, eta=ETA):
        g = torch.Generator(device="cpu").manual_seed(seed)
        # weights as normalised distributions per neuron
        w = torch.rand(n_exc, n_input, generator=g) * 0.3
        self.W = (w / w.sum(1, keepdim=True) * NORM_TARGET).to(device)
        self.theta = torch.zeros(n_exc, device=device)   # adaptive-threshold offset
        self.n_input, self.n_exc, self.device = n_input, n_exc, device
        self.theta_plus, self.eta = theta_plus, eta

    def present(self, rates, T, learn, noise_std=0.0):
        """Run one image for T timesteps. rates: [n_input] spike prob/timestep. Returns spike counts."""
        dev = self.device
        v = torch.zeros(self.n_exc, device=dev)
        x_pre = torch.zeros(self.n_input, device=dev)      # pre-synaptic traces
        counts = torch.zeros(self.n_exc, device=dev)
        rates = rates.to(dev)
        for _ in range(T):
            s_in = (torch.rand(self.n_input, device=dev) < rates).float()
            x_pre = x_pre * TR_DECAY + s_in
            I = self.W @ s_in                               # input current per exc neuron
            v = v * V_DECAY + I
            if noise_std > 0:
                v = v + torch.randn(self.n_exc, device=dev) * noise_std
            above = v - (V_THRESH + self.theta)
            if (above > 0).any():
                # single-winner WTA: strongest-driven neuron above threshold fires and, via
                # strong lateral inhibition, suppresses the rest for this timestep
                w = int(above.argmax())
                counts[w] += 1
                v[w] = V_RESET
                if learn:
                    self.theta[w] += self.theta_plus       # persistent homeostasis
                    # trace-STDP: on post spike, move that neuron's weights toward the pre-trace
                    self.W[w] += self.eta * (x_pre - self.W[w])
                    self.W[w].clamp_(0.0, W_MAX)
            if learn and THETA_DECAY < 1.0:
                self.theta *= THETA_DECAY
        if learn:
            # renormalise each neuron's input weights (competition)
            self.W *= (NORM_TARGET / self.W.sum(1, keepdim=True).clamp_min(1e-6))
        return counts

    def present_image(self, image, T, learn, noise_std=0.0, min_spikes=5, max_tries=6):
        """D&C intensity ramping: re-present at higher intensity until the image drives
        enough spikes. Ensures dim digits (thin '1's) still produce learning signal, fixing the
        class-representation imbalance where low-pixel digits get almost no dedicated neurons."""
        intensity = INTENSITY
        counts = None
        for _ in range(max_tries):
            counts = self.present(image * intensity, T, learn, noise_std)
            if counts.sum() >= min_spikes:
                break
            intensity += 0.10
        return counts


def load_mnist(n_train, n_test):
    tf = transforms.ToTensor()
    tr = datasets.MNIST("./data", train=True, download=True, transform=tf)
    te = datasets.MNIST("./data", train=False, download=True, transform=tf)
    Xtr = tr.data[:n_train].float().view(n_train, -1) / 255.0
    ytr = tr.targets[:n_train]
    Xte = te.data[:n_test].float().view(n_test, -1) / 255.0
    yte = te.targets[:n_test]
    return Xtr, ytr, Xte, yte


def assign_labels(net, X, y, T, n_classes=10):
    """Sum each neuron's response per class over a labelling set; label neuron by its best class."""
    resp = torch.zeros(net.n_exc, n_classes, device=net.device)
    for i in range(X.size(0)):
        counts = net.present_image(X[i], T, learn=False)
        resp[:, y[i]] += counts
    return resp.argmax(1)  # [n_exc] -> class label per neuron


def evaluate(net, X, y, neuron_label, T, n_classes=10):
    correct = 0
    total_spikes = 0.0
    silent = 0
    for i in range(X.size(0)):
        counts = net.present_image(X[i], T, learn=False)
        total_spikes += counts.sum().item()
        if counts.sum().item() == 0:
            silent += 1
        # class score = total spikes of neurons assigned to that class
        scores = torch.zeros(n_classes, device=net.device)
        scores.index_add_(0, neuron_label, counts)
        if scores.argmax().item() == y[i].item():
            correct += 1
    print(f"  eval: {total_spikes/X.size(0):.1f} spikes/img, {silent} silent imgs")
    return correct / X.size(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-exc", type=int, default=400)
    ap.add_argument("--n-train", type=int, default=15000)
    ap.add_argument("--n-label", type=int, default=10000)  # labelling set (subset of train)
    ap.add_argument("--n-test", type=int, default=10000)
    ap.add_argument("--T", type=int, default=80)
    ap.add_argument("--passes", type=int, default=1)
    ap.add_argument("--noise-std", type=float, default=0.0)
    ap.add_argument("--theta-plus", type=float, default=THETA_PLUS)
    ap.add_argument("--eta", type=float, default=ETA)
    ap.add_argument("--device", default="cpu")  # tiny per-step tensors: CPU beats MPS launch overhead
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    print(f"device: {device} | n_exc: {args.n_exc} | n_train: {args.n_train} | T: {args.T} "
          f"| noise: {args.noise_std}")

    Xtr, ytr, Xte, yte = load_mnist(args.n_train, args.n_test)
    net = STDPNet(784, args.n_exc, device, seed=args.seed,
                  theta_plus=args.theta_plus, eta=args.eta)

    # --- unsupervised training (no labels) ---
    t0 = time.time()
    for p in range(args.passes):
        for i in range(args.n_train):
            net.present_image(Xtr[i], args.T, learn=True, noise_std=args.noise_std)
            if (i + 1) % 2000 == 0:
                dead = (net.W.sum(1) < 1e-3).sum().item()
                print(f"  pass {p+1} img {i+1}/{args.n_train}  "
                      f"({time.time()-t0:.0f}s)  theta[min/max] "
                      f"{net.theta.min():.2f}/{net.theta.max():.2f}  dead {dead}")

    # --- label neurons (readout) then test ---
    nlab = min(args.n_label, args.n_train)
    neuron_label = assign_labels(net, Xtr[:nlab], ytr[:nlab], args.T)
    per_class = torch.bincount(neuron_label, minlength=10).tolist()
    print(f"neurons per class: {per_class}")
    acc = evaluate(net, Xte, yte, neuron_label, args.T)

    print("\n" + "=" * 48)
    print(f"  unsupervised STDP  |  {args.n_exc} neurons  |  noise {args.noise_std}")
    print(f"  test accuracy: {acc*100:.2f}%   (surrogate-gradient ref: 98.9%)")
    print("=" * 48)


if __name__ == "__main__":
    main()
