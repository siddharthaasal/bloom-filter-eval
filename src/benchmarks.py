"""
Main benchmarking script for Bloom Filter evaluation.
Runs experiments across different filters and workloads, collecting metrics.
"""

import time
import os
import csv
import pandas as pd
import numpy as np
import config
import workloads
from filters import (
    StandardBloomFilter,
    ScalableBloomFilter,
    CountingBloomWrapper,
    CuckooFilterWrapper,
)

# Ensure results directories exist
os.makedirs(config.CSV_DIR, exist_ok=True)
os.makedirs(config.PLOTS_DIR, exist_ok=True)


def measure_fpr(filter_obj, true_negatives):
    """Calculates False Positive Rate."""
    if len(true_negatives) == 0:
        return 0.0

    false_positives = 0
    # pybloom_live filters need string or byte inputs usually, or at least hashable.
    # Our keys are integers. They should be fine.

    for key in true_negatives:
        if key in filter_obj:
            false_positives += 1

    return false_positives / len(true_negatives)


def run_trial(run_id, filter_cls, workload_func, workload_params):
    """Runs a single trial for a specific filter and workload."""

    # 1. Setup Filter
    # Instantiate filter
    try:
        f = filter_cls(capacity=config.FILTER_CAPACITY, error_rate=config.ERROR_RATE)
    except Exception as e:
        print(f"Skipping {filter_cls.__name__}: {e}")
        return None

    # 2. Setup Workload
    # params is a dict, e.g., {'alpha': 1.0}
    insert_keys, query_keys, true_negatives = workload_func(
        num_keys=config.NUM_KEYS, seed=config.SEED + run_id, **workload_params
    )

    # 3. Measure Insert Time
    start_time = time.perf_counter()
    for key in insert_keys:
        f.add(key)
    end_time = time.perf_counter()
    insert_duration = end_time - start_time
    avg_insert_latency = (insert_duration / len(insert_keys)) * 1e6  # microseconds

    # 4. Measure Query Time
    start_time = time.perf_counter()
    for key in query_keys:
        _ = key in f
    end_time = time.perf_counter()
    query_duration = end_time - start_time
    avg_query_latency = (query_duration / len(query_keys)) * 1e6  # microseconds

    # 5. Measure FPR
    fpr = measure_fpr(f, true_negatives)

    # 6. Measure Memory
    # bits per key (approximate relative to items inserted)
    total_bits = f.size_bits()
    bits_per_key = total_bits / len(insert_keys) if len(insert_keys) > 0 else 0

    return {
        "RunID": run_id,
        "Filter": f.name,
        "Workload": workload_func.__name__,
        "Params": str(workload_params),
        "InsertLatency(us)": avg_insert_latency,
        "QueryLatency(us)": avg_query_latency,
        "FPR": fpr,
        "BitsPerKey": bits_per_key,
    }


def main():
    print("Starting Benchmarks...")
    print(f"Keys: {config.NUM_KEYS}, Trials: {config.NUM_TRIALS}")

    results = []

    filters_to_test = [
        StandardBloomFilter,
        ScalableBloomFilter,
        CountingBloomWrapper,
        CuckooFilterWrapper,
    ]

    # filters_to_test = [StandardBloomFilter] # Debugging: Start small

    # Define experiments
    experiments = []

    # 1. Uniform Workload
    experiments.append((workloads.generate_uniform, {}))

    # 2. Zipfian Workloads
    for alpha in config.ZIPF_ALPHAS:
        experiments.append((workloads.generate_zipfian, {"alpha": alpha}))

    for workload_func, params in experiments:
        print(f"\nWorkload: {workload_func.__name__} {params}")

        for filter_cls in filters_to_test:
            print(f"  Testing {filter_cls.__name__}...", end="", flush=True)

            for i in range(config.NUM_TRIALS):
                res = run_trial(i, filter_cls, workload_func, params)
                if res:
                    results.append(res)
                    print(".", end="", flush=True)
            print(" Done.")

    # Save Results
    df = pd.DataFrame(results)
    output_file = f"{config.CSV_DIR}/benchmark_results.csv"
    df.to_csv(output_file, index=False)

    print(f"\nResults saved to {output_file}")

    # Basic Summary
    summary = df.groupby(["Filter", "Workload", "Params"]).mean(numeric_only=True)
    print("\nSummary Results:")
    print(summary[["InsertLatency(us)", "QueryLatency(us)", "FPR", "BitsPerKey"]])


if __name__ == "__main__":
    main()
