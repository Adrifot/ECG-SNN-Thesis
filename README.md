# ECG Analysis using Spiking Neural Networks in Julia

Spiking Neural Networks (SNNs) with Spike-Timing-Dependent Plasticity (STDP), implemented from scratch in Julia, for detecting myocardial infarction (MI) in ECG signals.

This is the codebase for my Bachelor's thesis at the University of Bucharest, Faculty of Mathematics and Computer Science. It implements a full pipeline: raw ECG → band-pass filtering → beat segmentation → delta-modulation spike encoding → a routed, multi-lead STDP network → a linear (LDA) readout, and evaluates it on the PTB Diagnostic ECG Database.

**Headline result:** a segment-routed, multi-lead architecture (P/QRS/T windows × per-lead concatenation) reaches a held-out test **AUROC of 0.851** (95% CI [0.779, 0.912], MCC 0.572) on the PTB database, confirming that STDP-adapted per-neuron firing rates may carry clinically relevant information.

## Demo

`examples/demo_healthy_vs_infarction.jl` takes the exact winning network configuration and runs it with identical initial weights on one healthy and one infarction recording, side by side recording the delta-modulation input encoding, the hidden/output spike rasters, and the hidden→output synaptic weight matrix as they evolve under STDP:

![Healthy vs. infarction network dynamics](docs/imgs/demo_healthy_vs_infarction.gif)

This is purely illustrative of the encoding + STDP dynamics: the *before* stage, prior to feature extraction and LDA scoring. It is not a classifier or detector on its own; for the scored pipeline see [`confirm_routed_multilead.jl`](confirm_routed_multilead.jl) and [`examples/eval_routed_multilead.jl`](examples/eval_routed_multilead.jl).

Regenerate it with:

```bash
julia --project=. examples/demo_healthy_vs_infarction.jl
# or
./run_all.sh demo
```

## Repository structure

```
modules/          Core library: neurons, synapses, layers/network, signal processing, metrics
examples/         Runnable example & evaluation scripts (includes the demo above)
figures/          Scripts that regenerate the plots used in the thesis, plus a feature cache
docs/             Generated images (docs/imgs/) and the LaTeX thesis source (docs/latex/, not tracked)
ecg-db/           PTB Diagnostic ECG Database (not tracked, see Setup below)
paramsearch.jl, paramsearch_routed_multilead.jl 
                  Hyperparameter search scripts for the two network architectures
confirm_routed_multilead.jl 
               Robustness confirmation for the result (fixed config, varying splits)
run_all.sh   Convenience runner (see Usage below)
```

`modules/` is the actual library and has no dependency on anything else in the repo:

| Module | Purpose |
|---|---|
| `Neurons.jl` | Leaky integrate-and-fire (LIF) neuron model and the `Spike` type |
| `Synapses.jl` | `Synapse` type and its STDP (LTP/LTD) update rules |
| `Layers.jl` | `NeuronLayer`, `SynapseLayer`, `LayeredNetwork`, and the simulation loop (`runlayers!`) |
| `Signals.jl` | ECG loading, band-pass filtering, R-peak detection, beat segmentation, delta-modulation |
| `Registry.jl` | Parses PTB `.hea` headers into labelled `PatientRecord`s (`:healthy` / `:infarction`) |
| `Metrics.jl` | Imbalance-aware classification metrics (MCC, AUROC, bootstrap/Wilson CIs, etc.) |
| `Utils.jl` | Weight-initialization distributions and parameter-noise helpers |

## Requirements

- Julia ≥ 1.12 
- The PTB Diagnostic ECG Database (see below). Everything else is installed with `Pkg`

## Setup

1. **Get the PTB Diagnostic ECG Database.** It's not included in this repo. Download it from PhysioNet (<https://physionet.org/content/ptbdb/1.0.0/>) and place its contents directly under `ecg-db/`, so that you end up with:

   ```
   ecg-db/RECORDS
   ecg-db/CONTROLS
   ecg-db/patient001/...
   ecg-db/patient002/...
   ...
   ```

2. **Install the Julia dependencies:**

   ```bash
   julia --project=. -e 'using Pkg; Pkg.instantiate()'
   # or
   ./run_all.sh setup
   ```

## Usage

`run_all.sh` wraps the common entry points (run `chmod +x run_all.sh` once if needed, or invoke
it as `bash run_all.sh` / `JULIA=julia.exe bash run_all.sh` on Windows):

```bash
./run_all.sh setup     # install Julia packages
./run_all.sh figures   # regenerate the fast, single-recording thesis figures
./run_all.sh demo      # regenerate the healthy-vs-infarction demo GIF
./run_all.sh results   # regenerate the classifier figures (SLOW)
./run_all.sh confirm   # reproduce the headline AUROC result (SLOW)
./run_all.sh all       # setup + figures + demo
```

**On "SLOW":** `results` and `confirm` run the full STDP simulation over dozens of ECG recordings per class (and, for `confirm`, across multiple data splits and lead sets). On a first run, before `figures/cache/` is populated; this can take from several minutes to about an hour depending on your machine. Subsequent figure-only reruns reuse the cache.

Any individual script can also be run directly with the `julia --project=. [script]` command.
