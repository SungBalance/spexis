# SPDX-License-Identifier: Apache-2.0
"""Saturated decode throughput of a tensor-parallel baseline run.

Reads the per-iteration JSONL log written with --enable-log, finds the
first window of --num-iters consecutive iterations in which the
scheduler runs the full batch, and reports the decode throughput over
that window.
"""
import argparse
import json
import sys


def parse_args():
    parser = argparse.ArgumentParser(
        description="Saturated throughput analysis (TP baseline run)")
    parser.add_argument("--temp-log-path", type=str, required=True,
                        help="Per-iteration JSONL log from the benchmark")
    parser.add_argument("--batch-size", type=int, required=True,
                        help="Number of requests in the fixed batch")
    parser.add_argument("--pp-size", type=int, default=1,
                        help="Unused; kept for interface parity with the "
                        "PP analyzer")
    parser.add_argument("--skip-iters", type=int, default=10,
                        help="Warmup iterations to skip before measuring")
    parser.add_argument("--num-iters", type=int, default=50,
                        help="Consecutive full-batch iterations to measure")
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
                num_tokens == args.batch_size:
            start_time = entry["time"]

        if start_time != 0:
            if num_tokens != args.batch_size:
                break
            generated_tokens += num_tokens
            num_iter += 1
            end_time = entry["time"]
            if num_iter == args.num_iters:
                break

    if num_iter == 0 or end_time <= start_time:
        print("[Error] No full-batch window found in the log "
              f"(expected {args.batch_size} scheduled tokens per iteration).")
        sys.exit(1)
    if num_iter != args.num_iters:
        print(f"[Warning] Only {num_iter} iterations were measured; "
              f"expected {args.num_iters}.")

    report([
        ("Measured iterations:", num_iter),
        ("Measured window (s):", f"{end_time - start_time:.3f}"),
        ("Generated tokens:", generated_tokens),
        ("Throughput (tokens/s):",
         f"{generated_tokens / (end_time - start_time):.4f}"),
    ])


if __name__ == "__main__":
    main()
