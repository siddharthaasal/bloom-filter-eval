"""
Main benchmarking script for Bloom Filter evaluation.
Runs experiments across different filters and workloads, collecting metrics.
Includes Deletion, Adversarial, and Temporal scenarios.
"""

import time
import os
import pandas as pd
import numpy as np
import config
import workloads
from filters import (
    StandardBloomFilter,
    ScalableBloomFilter,
    CustomCountingBloomFilter,
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
    for key in true_negatives:
        if key in filter_obj:
            false_positives += 1

    return false_positives / len(true_negatives)


def run_trial(run_id, filter_cls, workload_func, workload_params):
    """
    Runs a single trial for a specific filter and workload.
    Handles standard, adversarial, and temporal workloads.
    """

    # Special Handling for Temporal Workload (Multi-phase)
    if workload_func == workloads.generate_temporal:
        return run_temporal_trial(run_id, filter_cls, workload_func, workload_params)

    # 1. Setup Filter
    try:
        f = filter_cls(capacity=config.FILTER_CAPACITY, error_rate=config.ERROR_RATE)
        f_sim_for_adv = None

        # If Adversarial, we need to pass a trained filter of the SAME type to the generator
        if workload_func == workloads.generate_adversarial:
            f_sim_for_adv = filter_cls(
                capacity=config.FILTER_CAPACITY, error_rate=config.ERROR_RATE
            )

    except Exception as e:
        print(f"Skipping {filter_cls.__name__}: {e}")
        return None

    # 2. Setup Workload
    # Handle Adversarial special case signature
    if workload_func == workloads.generate_adversarial:
        insert_keys, query_keys, true_negatives = workload_func(
            num_keys=config.NUM_KEYS,
            seed=config.SEED + run_id,
            filter_sim=f_sim_for_adv,
            **workload_params,
        )
    else:
        insert_keys, query_keys, true_negatives = workload_func(
            num_keys=config.NUM_KEYS, seed=config.SEED + run_id, **workload_params
        )

    # 3. Measure Insert Time
    start_time = time.perf_counter()
    for key in insert_keys:
        f.add(key)
    end_time = time.perf_counter()
    insert_duration = end_time - start_time
    avg_insert_latency = (
        (insert_duration / len(insert_keys)) * 1e6 if len(insert_keys) > 0 else 0
    )

    # 4. Measure Query Time & Latency Distribution (Cache Proxy)
    latencies = []
    start_time = time.perf_counter()

    # We want individual latencies for P99.
    # Calling time.perf_counter() inside the loop adds significant overhead.
    # For a high-performance benchmark, this is tricky in Python.
    # Strategy: Run the batch for Average. Run a smaller sample for Tail Latency?
    # Or just accept the overhead for "Research" precision.
    # Let's run a subset for P99 to avoid slowing down the main query loop too much if N is huge.
    # But for N=1000 it's fine.

    # Optimization: perform the main bulk query first for "Avg" (pure speed).
    # Then run a separate sample pass for "P99" (instrumented).

    # Fast pass
    for key in query_keys:
        _ = key in f
    end_time = time.perf_counter()
    query_duration = end_time - start_time
    avg_query_latency = (
        (query_duration / len(query_keys)) * 1e6 if len(query_keys) > 0 else 0
    )

    # Instrumented pass (Sample 10% or max 1000 keys)
    sample_size = min(len(query_keys), 1000)
    if sample_size > 0:
        sample_keys = query_keys[:sample_size]
        sample_lats = []
        for key in sample_keys:
            t0 = time.perf_counter()
            _ = key in f
            t1 = time.perf_counter()
            sample_lats.append(t1 - t0)
        p99_latency_us = np.percentile(sample_lats, 99) * 1e6
    else:
        p99_latency_us = 0

    # 5. Measure FPR
    fpr = measure_fpr(f, true_negatives)

    # 6. Measure Memory
    total_bits = f.size_bits()
    bits_per_key = total_bits / len(insert_keys) if len(insert_keys) > 0 else 0

    # 7. Measure Deletion (if applicable)
    avg_delete_latency = 0
    try:
        # Try to delete 10% of keys
        keys_to_delete = insert_keys[: int(len(insert_keys) * 0.1)]
        if keys_to_delete.size > 0:
            start_del = time.perf_counter()
            for key in keys_to_delete:
                f.delete(key)
            end_del = time.perf_counter()
            avg_delete_latency = ((end_del - start_del) / len(keys_to_delete)) * 1e6
    except NotImplementedError:
        avg_delete_latency = None  # Indicator for not supported
    except Exception:
        avg_delete_latency = None

    return {
        "RunID": run_id,
        "Filter": f.name,
        "Workload": workload_func.__name__,
        "Params": str(workload_params),
        "InsertLatency(us)": avg_insert_latency,
        "QueryLatency(us)": avg_query_latency,
        "P99_QueryLatency(us)": p99_latency_us,
        "DeleteLatency(us)": avg_delete_latency,
        "FPR": fpr,
        "BitsPerKey": bits_per_key,
    }


def run_temporal_trial(run_id, filter_cls, workload_func, workload_params):
    """
    Runs a temporal trial with phases.
    Returns a list of result dicts (one per phase).
    """
    results = []
    try:
        f = filter_cls(capacity=config.FILTER_CAPACITY, error_rate=config.ERROR_RATE)
    except Exception:
        return []

    cycle_data = workload_func(
        num_keys=config.NUM_KEYS, seed=config.SEED + run_id, **workload_params
    )

    for phase_data in cycle_data:
        phase = phase_data["phase"]
        insert_keys = phase_data["insert_keys"]
        query_keys = phase_data["query_keys"]

        # Insert
        start_time = time.perf_counter()
        for key in insert_keys:
            f.add(key)
        end_time = time.perf_counter()
        insert_lat = (end_time - start_time) / len(insert_keys) * 1e6

        # Query
        start_time = time.perf_counter()
        for key in query_keys:
            _ = key in f
        end_time = time.perf_counter()
        query_lat = (end_time - start_time) / len(query_keys) * 1e6

        # Query P99 (Sample)
        sample_keys = query_keys[: min(len(query_keys), 100)]
        lats = []
        for k in sample_keys:
            t0 = time.perf_counter()
            _ = k in f
            lats.append(time.perf_counter() - t0)
        p99_lat = np.percentile(lats, 99) * 1e6 if lats else 0

        # FPR Calculation
        true_negatives = phase_data.get("true_negatives", [])
        fpr = 0.0
        if len(true_negatives) > 0:
            fpr = measure_fpr(f, true_negatives)

        results.append(
            {
                "RunID": run_id,
                "Filter": f.name,
                "Workload": "Temporal",
                "Params": f"Phase={phase}",
                "InsertLatency(us)": insert_lat,
                "QueryLatency(us)": query_lat,
                "P99_QueryLatency(us)": p99_lat,
                "DeleteLatency(us)": None,  # Not focused in temporal yet
                "FPR": fpr,
                "BitsPerKey": f.size_bits()
                / (len(insert_keys) * (phase + 1)),  # Rough approx
            }
        )

    return results


def main():
    print("Starting Research Benchmarks...")
    print(f"Keys: {config.NUM_KEYS}, Trials: {config.NUM_TRIALS}")

    results = []

    filters_to_test = [
        StandardBloomFilter,
        ScalableBloomFilter,
        CustomCountingBloomFilter,
        CuckooFilterWrapper,
    ]

    # Define experiments
    experiments = []

    # 1. Uniform
    experiments.append((workloads.generate_uniform, {}))

    # 2. Zipfian
    for alpha in config.ZIPF_ALPHAS:
        experiments.append((workloads.generate_zipfian, {"alpha": alpha}))

    # 3. Adversarial
    experiments.append(
        (workloads.generate_adversarial, {"attack_ratio": config.ADVERSARIAL_RATIO})
    )

    # 4. Temporal
    experiments.append(
        (workloads.generate_temporal, {"phases": config.TEMPORAL_PHASES})
    )

    for workload_func, params in experiments:
        print(f"\nWorkload: {workload_func.__name__} {params}")

        for filter_cls in filters_to_test:
            print(f"  Testing {filter_cls.__name__}...", end="", flush=True)

            for i in range(config.NUM_TRIALS):
                res = run_trial(i, filter_cls, workload_func, params)
                if res:
                    if isinstance(res, list):
                        results.extend(res)
                    else:
                        results.append(res)
                    print(".", end="", flush=True)
            print(" Done.")

    # Save Results
    df = pd.DataFrame(results)
    output_file = f"{config.CSV_DIR}/research_benchmark_results.csv"
    df.to_csv(output_file, index=False)

    print(f"\nResults saved to {output_file}")

    # Research Summary
    summary = df.groupby(["Filter", "Workload", "Params"]).mean(numeric_only=True)
    print("\nSummary Results (Preview):")
    # Show cols of interest
    cols = [
        "InsertLatency(us)",
        "QueryLatency(us)",
        "DeleteLatency(us)",
        "FPR",
        "BitsPerKey",
    ]
    print(summary[cols].to_string())


if __name__ == "__main__":
    main()
