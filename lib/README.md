# lib/ — third-party tools (not vendored)

## NeuroTrain (fetched 2026-10-20)

Open benchmarking framework for SNN training algorithms (Caviglia et al., arXiv 2605.15058; Politecnico di Torino). snnTorch + Tonic + NeuroBench; 11 registered trainers (bptt, decolle, ell, eprop, esd_rtrl, etlp, ostl, osttp, ottt, stsf, tp); datasets MNIST, Fashion-MNIST, CIFAR-10, SVHN, N-MNIST, DVS-Gesture, DVS-CIFAR10, SHD.

Fetch (git clone is blocked inside the Claude Code sandbox; the tarball is not):

    cd lib && curl -sfL https://github.com/smilies-polito/neurotrain/archive/refs/heads/main.tar.gz | tar xz && mv neurotrain-main neurotrain

Deps (already in `.venv` as of 2026-10-20; torch/snntorch untouched): tonic, neurobench, aedat, optuna, optuna-dashboard, pyyaml, tensorboard, matplotlib, tqdm.

Run (GPU needs the sandbox lifted; MNIST already in `../data`):

    cd lib/neurotrain && ../../.venv/bin/python run_exp_campaign.py --custom config/custom/smoke_mps.yaml --name smoke_mps --no-progress

Config notes for this machine: `device: mps`, `num_workers: 0`, `data_root: ../../data`. Our own configs live in `lib/neurotrain/config/custom/` and are copied into `experiments/neurotrain_configs/` when they matter.
