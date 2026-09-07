# Experiments

Scripts to reproduce the paper's main results. Both scripts are
self-contained: every knob is an environment variable with a sensible
default, listed at the top of each script.

## Prerequisites

- Spexis installed (see the top-level README) on a multi-GPU node.
- The target model weights (e.g. `Llama-3.3-70B-Instruct`).
- The exit-model artifacts for that model (`exit_1_2/`, `exit_1_4/`,
  `embed_tokens.bin`), and for `SPEXIS_VER=2.5` the length-predictor
  weights.
- A dataset file. The paper mainly uses
  [SpecBench](https://github.com/hemingkx/Spec-Bench)
  (`data/spec_bench/question.jsonl`); `sharegpt`, `ultrachat` and
  `lmsyschat` are also supported by the harness.

## 1. Serving experiment (`serving/run_serving.sh`)

The main end-to-end comparison: an OpenAI-compatible server is launched
for each configuration, and a client submits requests as a Poisson
process at a set of request rates for a fixed duration.

```bash
cd experiments/serving
MODEL_DIR=/models/Llama-3.3-70B-Instruct \
EXIT_MODEL_DIR=/models/exitmodels/Llama-3.3-70B-Instruct \
DATASET_PATH=/data/spec_bench/question.jsonl \
PP_TP_PAIRS="2,4" SPEXIS_OPTS="True False" SPEXIS_VER=1.0 \
REQUEST_RATES="3.8 4.0 4.2" \
./run_serving.sh
```

Each run writes `client.log` (latency/throughput summary),
`server_console.log`, and the per-iteration server log
(`server_iter_log.jsonl`) under `results/serving/<experiment-id>/`.
The baseline runs (`SPEXIS_OPTS` containing `False`) use plain PP/TP
with the same engine settings.

## 2. Fixed-batch throughput experiment (`throughput/run_fixed_batch.sh`)

Measures saturated decode throughput at a fixed batch size N: all N
requests enter the engine at once with EOS ignored, so the running
batch stays at N, and `log_iter/throughput_saturated*.py` measures
tokens/s over a window of 50 consecutive full-batch iterations from the
per-iteration log (for Spexis runs it also reports the speculation
accuracy).

```bash
cd experiments/throughput
MODEL_DIR=/models/Llama-3.3-70B-Instruct \
EXIT_MODEL_DIR=/models/exitmodels/Llama-3.3-70B-Instruct \
DATASET_PATH=/data/spec_bench/question.jsonl \
PP_TP_PAIRS="2,1" BATCH_SIZES="16 32 64" \
./run_fixed_batch.sh
```

Notes:

- With `DROP_RATIO > 0` the analyzer measures the window at the
  effective batch size `N - floor(N * drop_ratio)`.
- With `SPEXIS_VER=2.5` the length-aware boost can occasionally
  schedule more than the effective batch size, so the strict full-batch
  window may be shorter than 50 iterations; the analyzer prints a
  warning when that happens.

## Layout

```
experiments/
  benchmark/    client and offline harnesses (serving duration, throughput)
  log_iter/     per-iteration-log analyzers (saturated throughput)
  serving/      experiment 1
  throughput/   experiment 2
```
