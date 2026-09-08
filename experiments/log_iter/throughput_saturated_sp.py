# SPDX-License-Identifier: Apache-2.0
"""Saturated decode throughput of a Spexis run.

Reads the per-iteration JSONL log written with --enable-log, finds the
first window of --num-iters consecutive iterations in which the
scheduler runs the full (drop-adjusted) batch, and reports the decode
throughput over that window together with speculation statistics.
"""
import argparse
import json
import sys


def parse_args():
    parser = argparse.ArgumentParser(
        description="Saturated throughput analysis (Spexis run)")
    parser.add_argument("--temp-log-path", type=str, required=True,
                        help="Per-iteration JSONL log from the benchmark")
    parser.add_argument("--batch-size", type=int, required=True,
                        help="Number of requests in the fixed batch")
    parser.add_argument("--drop-ratio", type=float, default=0.0,
                        help="Drop ratio the run used")
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
    # The measurement window matches the batch the scheduler actually
    # runs: the drop keeps only the top (1 - drop_ratio) fraction.
    eff_bs = args.batch_size - int(args.batch_size * args.drop_ratio)

    correct_total = 0
    wrong_total = 0
    flush_total = 0
    drop_total = 0
    cpu_emb_times = []

    generated_tokens = 0
    start_time = 0
    end_time = 0
    num_iter = 0

    for entry in iter_log(args.temp_log_path):
        spec = entry.get("spec", {})
        correct = spec.get("correct", 0)
        wrong = spec.get("wrong", 0)
        correct_total += correct
        wrong_total += wrong
        flush_total += spec.get("flush_norm", 0) + spec.get("flush_drop", 0)
        drop_total += spec.get("final_drop", 0)

        cpu_emb = entry.get("cpu_embed_s")
        if cpu_emb:
            cpu_emb_times.append(cpu_emb["update_states"] +
                                 cpu_emb["reorder"] + cpu_emb["lookup"])

        num_tokens = entry["sched"]["tokens"]
        if num_tokens == 0:
            continue

        if num_tokens == eff_bs and start_time == 0:
            start_time = entry["time"]

        if start_time != 0:
            if num_tokens != eff_bs:
                break
            generated_tokens += correct + wrong
            num_iter += 1
            end_time = entry["time"]
            if num_iter == args.num_iters:
                break

    if num_iter == 0 or end_time <= start_time:
        print("[Error] No full-batch window found in the log "
              f"(expected {eff_bs} scheduled tokens per iteration).")
        sys.exit(1)
    if num_iter != args.num_iters:
        print(f"[Warning] Only {num_iter} iterations were measured; "
              f"expected {args.num_iters}.")

    verified = correct_total + wrong_total
    all_spec = verified + flush_total + drop_total
    avg_cpu_emb = sum(cpu_emb_times) / len(cpu_emb_times) \
        if cpu_emb_times else 0.0

    report([
        ("Measured iterations:", num_iter),
        ("Measured window (s):", f"{end_time - start_time:.3f}"),
        ("Generated tokens:", generated_tokens),
        ("Throughput (tokens/s):",
         f"{generated_tokens / (end_time - start_time):.4f}"),
        ("Spec accuracy (%):",
         f"{correct_total / verified * 100:.2f}" if verified else "n/a"),
        ("Spec dropped (%):",
         f"{drop_total / all_spec * 100:.2f}" if all_spec else "n/a"),
        ("CPU embedding (ms/iter):", f"{avg_cpu_emb * 1000:.3f}"),
    ])


if __name__ == "__main__":
    main()
