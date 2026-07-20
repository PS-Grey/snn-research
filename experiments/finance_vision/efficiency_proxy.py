"""Efficiency proxy: the SNN's real advantage isn't accuracy, it's events/inference and latency.

We can't measure on-chip latency/power on a Mac (simulation is SLOWER than a GPU CNN; the win is
a hardware property). But we can measure the proxy that determines it: spikes and synaptic
operations (SynOps) per inference, from which theoretical on-chip latency and energy follow.

Reports the trained flow-SNN's per-inference cost. Interpret against a GPU CNN qualitatively
(ms / mJ, dense MACs) -- exact CNN MACs would need the finance CNN definition.

Usage: .venv/bin/python experiments/finance_vision/efficiency_proxy.py
"""

import time
import numpy as np
import torch

from snn_chart import ChartSNN, load, run_epoch
from torch.utils.data import DataLoader, TensorDataset

# published rough figures (order-of-magnitude, for the estimate only)
LOIHI_PJ_PER_SYNOP = 20e-12      # ~pJ/SynOp, event-driven accumulate
LOIHI_US_PER_STEP = 5e-6         # ~us per algorithmic timestep
GPU_PJ_PER_MAC = 300e-12         # system-level (incl. data movement) for a small model, rough


def main():
    dev = torch.device("cpu")   # count is device-independent; CPU avoids MPS overhead here
    torch.manual_seed(0)
    (Xtr, ytr), (Xte, yte, ids, fwd) = load(flow=True, encoding="naive")
    F = Xtr.shape[-1]
    trl = DataLoader(TensorDataset(torch.tensor(Xtr), torch.tensor(ytr)), batch_size=128, shuffle=True)
    m = ChartSNN(n_feat=F, hidden=64).to(dev)
    opt = torch.optim.Adam(m.parameters(), lr=2e-3)
    print("training briefly for realistic spike rates (10 epochs)...")
    for _ in range(10):
        run_epoch(m, trl, dev, opt)

    # instrumented inference: count per-layer spikes on the test set
    m.eval()
    T = 64
    s1_tot = s2_tot = 0.0
    n = 0
    xb = torch.tensor(Xte[:1024])
    with torch.no_grad():
        mm1 = m.lif1.init_leaky(); mm2 = m.lif2.init_leaky(); mo = m.lif_out.init_leaky()
        for t in range(xb.size(1)):
            c1 = m.bn1(m.fc1(xb[:, t])); s1, mm1 = m.lif1(c1, mm1)
            c2 = m.bn2(m.fc2(s1)); s2, mm2 = m.lif2(c2, mm2)
            _, mo = m.lif_out(m.fc_out(s2), mo)
            s1_tot += float(s1.sum()); s2_tot += float(s2.sum())
    B = xb.size(0)
    s1_pi, s2_pi = s1_tot / B, s2_tot / B           # spikes per inference, per layer
    spikes_pi = s1_pi + s2_pi

    # SynOps per inference: fan-out per spike (fc2: 64, fc_out: 1); fc1 input is DENSE (not event)
    fanout1, fanout2 = 64, 1
    synops_event = s1_pi * fanout1 + s2_pi * fanout2
    dense_input_macs = F * 64 * T                    # fc1 dense every timestep (input coding)
    total_ops = synops_event + dense_input_macs

    latency_us = T * LOIHI_US_PER_STEP * 1e6
    energy_nj = (synops_event * LOIHI_PJ_PER_SYNOP + dense_input_macs * LOIHI_PJ_PER_MAC
                 if False else (synops_event * LOIHI_PJ_PER_SYNOP) * 1e9)
    # (dense input MACs also cost energy; keep the event-part headline, note dense separately)

    print("\n" + "=" * 60)
    print(f"  SNN flow model — efficiency profile (per inference)")
    print(f"  hidden 64, {T} timesteps, {F} input features")
    print("-" * 60)
    print(f"  spikes/inference      : {spikes_pi:8.0f}   (lif1 {s1_pi:.0f} + lif2 {s2_pi:.0f})")
    print(f"  event SynOps/inference: {synops_event:8.0f}   (only active neurons compute)")
    print(f"  dense input MACs      : {dense_input_macs:8.0f}   (fc1, NOT event-driven)")
    print(f"  sparsity (lif1 firing): {s1_pi/(64*T)*100:5.1f}% of neuron-steps")
    print("-" * 60)
    print(f"  theoretical on-chip (Loihi-class estimate):")
    print(f"    latency ~ {latency_us:.0f} us   ({T} steps x ~{LOIHI_US_PER_STEP*1e6:.0f} us)")
    print(f"    event energy ~ {energy_nj:.2f} nJ   (SynOps x ~20 pJ)")
    print(f"  contrast: a GPU CNN inference ~ single-digit ms, ~mJ (dense MACs + data movement)")
    print("=" * 60)
    print("caveat: estimate from published chip figures, not a measurement; the dense input-coding")
    print("path (fc1) is NOT event-driven -- a spike-coded input would cut it further.")


if __name__ == "__main__":
    main()
