"""Modern gradient-SNN baseline vs a matched ANN — the clean current SNN<->ANN gap.

Re-baselines the project under the new direction: surrogate-gradient training, SoTA-style
conv-BN-LIF net, NO membrane noise (the legacy 80.7 % MNIST figure was the old noise regime).
The SNN and ANN share an identical conv backbone and parameter count so the gap is
architecture-fair, not a capacity artefact.

Carried mechanisms kept: BatchNorm (non-optional), subtract-style reset.
Dropped for this regime: membrane noise.

Device: MPS (M2) when available, else CUDA, else CPU.

Usage:
    .venv/bin/python experiments/baseline_gradient_snn.py --dataset mnist --epochs 5
    .venv/bin/python experiments/baseline_gradient_snn.py --dataset cifar10 --epochs 30
"""

import argparse
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import snntorch as snn
from snntorch import surrogate


def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# --------------------------------------------------------------------------- #
# Models                                                                       #
# --------------------------------------------------------------------------- #
class ConvSNN(nn.Module):
    """Conv-BN-LIF backbone, direct input coding over T steps, membrane-accumulation readout."""

    def __init__(self, in_ch: int, n_classes: int, img_size: int, beta: float = 0.9, T: int = 10):
        super().__init__()
        self.T = T
        spike_grad = surrogate.atan()
        lif = lambda: snn.Leaky(beta=beta, spike_grad=spike_grad, reset_mechanism="subtract")

        self.conv1 = nn.Conv2d(in_ch, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.lif1 = lif()
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.lif2 = lif()
        feat = (img_size // 4) ** 2 * 64
        self.fc = nn.Linear(feat, n_classes)
        # readout LIF accumulates membrane, never resets within the window
        self.lif_out = snn.Leaky(beta=beta, spike_grad=spike_grad,
                                 reset_mechanism="none", output=True)

    def forward(self, x):
        m1 = self.lif1.init_leaky()
        m2 = self.lif2.init_leaky()
        m_out = self.lif_out.init_leaky()
        spike_count = 0.0
        mem_sum = 0.0
        for _ in range(self.T):
            c1 = F.max_pool2d(self.bn1(self.conv1(x)), 2)
            s1, m1 = self.lif1(c1, m1)
            c2 = F.max_pool2d(self.bn2(self.conv2(s1)), 2)
            s2, m2 = self.lif2(c2, m2)
            out = self.fc(s2.flatten(1))
            _, m_out = self.lif_out(out, m_out)
            mem_sum = mem_sum + m_out
            spike_count = spike_count + s1.sum() + s2.sum()
        # mean spikes per image across the whole window (proxy SynOps)
        self.last_spikes_per_img = spike_count / x.size(0)
        return mem_sum / self.T


class ConvANN(nn.Module):
    """Identical backbone, ReLU activations, single forward pass. Param-matched to ConvSNN."""

    def __init__(self, in_ch: int, n_classes: int, img_size: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        feat = (img_size // 4) ** 2 * 64
        self.fc = nn.Linear(feat, n_classes)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.bn1(self.conv1(x))), 2)
        x = F.max_pool2d(F.relu(self.bn2(self.conv2(x))), 2)
        return self.fc(x.flatten(1))


# --------------------------------------------------------------------------- #
# Train / eval                                                                 #
# --------------------------------------------------------------------------- #
def run_epoch(model, loader, device, opt=None):
    train = opt is not None
    model.train(train)
    total, correct, loss_sum = 0, 0, 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with torch.set_grad_enabled(train):
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            if train:
                opt.zero_grad()
                loss.backward()
                opt.step()
        loss_sum += loss.item() * x.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += x.size(0)
    return correct / total, loss_sum / total


def get_data(dataset: str, batch: int):
    if dataset == "mnist":
        tf = transforms.Compose([transforms.ToTensor(),
                                 transforms.Normalize((0.1307,), (0.3081,))])
        tr = datasets.MNIST("./data", train=True, download=True, transform=tf)
        te = datasets.MNIST("./data", train=False, download=True, transform=tf)
        return tr, te, 1, 10, 28
    elif dataset == "cifar10":
        tf_tr = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])
        tf_te = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])
        tr = datasets.CIFAR10("./data", train=True, download=True, transform=tf_tr)
        te = datasets.CIFAR10("./data", train=False, download=True, transform=tf_te)
        return tr, te, 3, 10, 32
    raise ValueError(dataset)


def n_params(m):
    return sum(p.numel() for p in m.parameters())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["mnist", "cifar10"], default="mnist")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--T", type=int, default=10)
    ap.add_argument("--beta", type=float, default=0.9)
    args = ap.parse_args()

    device = pick_device()
    torch.manual_seed(0)
    print(f"device: {device} | dataset: {args.dataset} | epochs: {args.epochs} | T: {args.T}")

    tr, te, in_ch, n_classes, img = get_data(args.dataset, args.batch)
    # num_workers=0: avoids macOS/MPS shared-memory manager timeouts; these datasets are small
    tr_loader = DataLoader(tr, batch_size=args.batch, shuffle=True, num_workers=0)
    te_loader = DataLoader(te, batch_size=256, shuffle=False, num_workers=0)

    results = {}
    for name, build in [
        ("ANN", lambda: ConvANN(in_ch, n_classes, img)),
        ("SNN", lambda: ConvSNN(in_ch, n_classes, img, beta=args.beta, T=args.T)),
    ]:
        model = build().to(device)
        opt = torch.optim.Adam(model.parameters(), lr=args.lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
        print(f"\n=== {name} ({n_params(model):,} params) ===")
        best = 0.0
        for ep in range(1, args.epochs + 1):
            t0 = time.time()
            tr_acc, tr_loss = run_epoch(model, tr_loader, device, opt)
            te_acc, _ = run_epoch(model, te_loader, device)
            sched.step()
            best = max(best, te_acc)
            extra = ""
            if name == "SNN":
                extra = f" | spikes/img {model.last_spikes_per_img:.0f}"
            print(f"  ep {ep:2d}  train {tr_acc*100:5.2f}%  test {te_acc*100:5.2f}%"
                  f"  ({time.time()-t0:4.1f}s){extra}")
        results[name] = best
        print(f"  best {name} test: {best*100:.2f}%")

    gap = results["ANN"] - results["SNN"]
    print("\n" + "=" * 48)
    print(f"  ANN  best test : {results['ANN']*100:.2f}%")
    print(f"  SNN  best test : {results['SNN']*100:.2f}%")
    print(f"  SNN<->ANN gap  : {gap*100:.2f} pp")
    print("=" * 48)


if __name__ == "__main__":
    main()
