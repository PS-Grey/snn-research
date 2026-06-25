# Baseline results — gradient-SNN vs matched ANN

Pure surrogate-gradient SNN (conv-BN-LIF, atan surrogate, subtract reset, **no membrane
noise**) against a parameter-matched ANN sharing an identical conv backbone. The point is a
fair SNN↔ANN gap under modern gradient training, replacing the legacy membrane-noise regime.

Script: `baseline_gradient_snn.py`. Device: MPS (M2). Both models 50,378 params.

| Dataset | Epochs | T | ANN best | SNN best | Gap (pp) | SNN spikes/img |
|---|---|---|---|---|---|---|
| MNIST | 5 | 10 | 99.13% | 98.89% | 0.24 | ~22,700 |
| Fashion-MNIST | 10 | 10 | 92.30% | 92.41% | **−0.11** | ~26,000 |

## Timestep (T) sweep — accuracy vs energy (Fashion-MNIST, 10 epochs each)

Script: `timestep_sweep.py`.

| T | Test acc | Spikes/img |
|---|---|---|
| 1 | 91.60% | 1,892 |
| 2 | 90.85% | 5,078 |
| 4 | 92.15% | 9,604 |
| 8 | 92.43% | 19,498 |

Accuracy is flat across T (90.8–92.4%) while spikes scale ~linearly with T. T=1→T=8 buys
+0.83 pp for 10x the spikes. On a static frame there is no temporal structure to integrate
(extra steps just re-present the same image), so **T=1 is the deployment operating point**:
~1,900 spikes/img at 91.6%. The temporal dimension only earns its cost on genuinely temporal
input (DVS / event vision), which is where higher T should pay off.

## Reading

- The legacy figure of **80.7% as the best pure SNN on MNIST** is an artefact of the old
  regime (membrane noise + bio-inspired stack at small scale), **not a ceiling**. With
  surrogate gradients and BatchNorm from the start, the pure SNN sits within a quarter of a
  point of its ANN on MNIST and ties (edges ahead) on the harder Fashion-MNIST.
- Mechanism: the SNN tracks the ANN learning curve epoch-for-epoch, so the surrogate
  gradient propagates cleanly through the conv-BN-LIF stack. No dead-neuron starvation, no
  collapse. Spike rate self-sparsifies and stabilises without any noise or spike-rate loss.
- Conclusion so far: the SNN↔ANN gap is a property of the training regime, not of spiking
  computation. Membrane noise is not needed once you train with surrogate gradients + BN.
