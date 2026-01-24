"""
Workload generation module for Bloom Filter benchmarking.
Generates synthetic datasets for Uniform and Zipfian distributions.
"""

import numpy as np
import config


def generate_uniform(num_keys=config.NUM_KEYS, seed=config.SEED):
    """
    Generates a uniform workload.

    Returns:
        insert_keys (list): Keys to be inserted into the filter.
        query_keys (list): Keys to query (mix of present and absent).
        true_negatives (list): Keys known to be NOT in the filter (for FPR).
    """
    np.random.seed(seed)
    # Generate unique keys for insertion (using large range to minimize collisions)
    # Using 3x range to ensure enough space for non-colliding negatives
    pool_size = num_keys * 3
    all_keys = np.random.choice(range(pool_size * 10), pool_size, replace=False)

    insert_keys = all_keys[:num_keys]

    # For query_keys, we want a mix. Let's say 50% present, 50% absent.
    # But for FPR, we specifically want absent keys.
    # The benchmark can decide how to use them.
    # Here we return a set of pure negatives for FPR calculation.
    true_negatives = all_keys[num_keys : num_keys * 2]

    # Query set: same size as insert, random mix
    query_pool = np.concatenate([insert_keys, true_negatives])
    query_keys = np.random.choice(query_pool, num_keys, replace=False)

    return insert_keys, query_keys, true_negatives


def generate_zipfian(num_keys=config.NUM_KEYS, alpha=1.0, seed=config.SEED):
    """
    Generates a Zipfian (skewed) workload.

    Args:
        num_keys (int): Number of unique keys to generate.
        alpha (float): Skew parameter (a > 1).
        seed (int): Random seed.

    Returns:
        insert_keys (list): Keys to be inserted.
        query_keys (list): Keys to query (skewed distribution).
        true_negatives (list): Keys known to be NOT in the filter.
    """
    np.random.seed(seed)

    # Generate unique keys for insertion first (Uniformly distinctive)
    # We insert unique items, but QUERY them with skew.
    pool_size = num_keys * 3
    all_keys = np.random.choice(range(pool_size * 10), pool_size, replace=False)
    insert_keys = all_keys[:num_keys]
    true_negatives = all_keys[num_keys : num_keys * 2]

    # Generate Zipfian indices to sample from the insert_keys
    # Zipf distribution in numpy: numpy.random.zipf(a, size)
    # Returns values from 1 to infinity. We need to map them to indices 0..num_keys-1
    # Note: numpy zipf is basic. For bounded Zipf, we often map modulo or rejection sample.
    # Simple approach: z = np.random.zipf(alpha) - 1. If z >= num_keys, take z % num_keys

    # Note: numpy.random.zipf requires alpha > 1.
    # For alpha=1.0 (Zipf's law), we need a value slightly > 1 or use specific generator.
    # The user plan mentions alpha=1.0. Numpy's zipf param 'a' must be > 1.
    # We will use a commonly accepted approximation or simply ensure a > 1.001.

    effective_alpha = max(alpha, 1.001)

    # Generate raw zipf values
    zipf_values = np.random.zipf(effective_alpha, num_keys)

    # Map to indices within insert_keys range
    # We subtract 1 because Zipf starts at 1
    indices = (zipf_values - 1) % len(insert_keys)

    # Create the skewed query workload from keys strictly in the set
    query_keys_present = insert_keys[indices]

    # For the benchmark, we might want to query negatives too?
    # Usually skewed workloads test cache hits on positive queries.
    # But for FPR, we just need the true_negatives list.

    return insert_keys, query_keys_present, true_negatives
