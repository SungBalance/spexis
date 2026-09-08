# SPDX-License-Identifier: Apache-2.0
"""Saturated decode throughput of a pipeline-parallel baseline run.

Reads the per-iteration JSONL log written with --enable-log. A PP
baseline splits the batch into micro-batches, so after skipping the
warmup iterations the analyzer locks onto the repeating micro-batch
pattern (one entry per pipeline stage) and measures the decode
throughput until the pattern breaks.

NOTE: the pattern break is what ends the window, so when requests start
finishing, one perturbed pair (e.g. 64/64 -> 63/61) may slip into the
measurement before the mismatch is caught.
"""
import argparse
import json
import sys


def parse_args():
    parser = argparse.ArgumentParser(
        description="Saturated throughput analysis (PP baseline run)")
    parser.add_argument("--temp-log-path", type=str, required=True,
                        help="Per-iteration JSONL log from the benchmark")
    parser.add_argument("--batch-size", type=int, required=True,
                        help="Number of requests in the fixed batch")
    parser.add_argument("--pp-size", type=int, default=1,
                        help="Pipeline-parallel size of the run")
    parser.add_argument("--skip-iters", type=int, default=150,
                        help="Warmup iterations to skip before measuring")
    return parser.parse_args()


def iter_log(path):
    """Yield parsed records from a per-iteration JSONL log."""
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                print("[Warning] Skipping malformed JSON line: "
                      f"{line.strip()[:50]}...")


def report(rows):
    for name, value in rows:
        print(f"{name:<26} {value}")


def main():
    args = parse_args()

    micro_batch_sizes = []
    generated_tokens = 0
    start_time = 0
    end_time = 0
    num_iter = 0

    for entry in iter_log(args.temp_log_path):
        num_tokens = entry["sched"]["tokens"]
        if num_tokens == 0:
            continue

        # Skip the warmup/prefill iterations before starting the window.
        if entry["iter"] > args.skip_iters and start_time == 0 and \
                num_tokens <= args.batch_size / args.pp_size:
            start_time = entry["time"]

        if start_time != 0:
            if num_iter < args.pp_size:
                # Lock onto the repeating micro-batch pattern.
                micro_batch_sizes.append(num_tokens)
            elif micro_batch_sizes[num_iter % args.pp_size] != num_tokens:
                # The pattern broke (requests finished): stop measuring.
                break
            generated_tokens += num_tokens
            num_iter += 1
            end_time = entry["time"]

    if num_iter == 0 or end_time <= start_time:
        print("[Error] No saturated window found in the log.")
        sys.exit(1)
    if sum(micro_batch_sizes) != args.batch_size:
        print(f"[Warning] Micro-batch sizes {micro_batch_sizes} do not sum "
              f"to the batch size ({args.batch_size}).")

    report([
        ("Measured iterations:", num_iter),
        ("Measured window (s):", f"{end_time - start_time:.3f}"),
        ("Generated tokens:", generated_tokens),
        ("Throughput (tokens/s):",
         f"{generated_tokens / (end_time - start_time):.4f}"),
        ("Micro-batch sizes:", micro_batch_sizes),
    ])


if __name__ == "__main__":
    main()
