"""
Workload generation module for Bloom Filter benchmarking.
Generates synthetic datasets for Uniform, Zipfian, Adversarial, and Temporal distributions.
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

    effective_alpha = max(alpha, 1.001)

    # Generate raw zipf values
    zipf_values = np.random.zipf(effective_alpha, num_keys)

    # Map to indices within insert_keys range
    # We subtract 1 because Zipf starts at 1
    indices = (zipf_values - 1) % len(insert_keys)

    # Create the skewed query workload from keys strictly in the set
    query_keys_present = insert_keys[indices]

    return insert_keys, query_keys_present, true_negatives


def generate_adversarial(
    num_keys=config.NUM_KEYS, seed=config.SEED, attack_ratio=0.5, filter_sim=None
):
    """
    Generates an Adversarial workload.

    This requires a pre-trained filter (or similar logic) to find False Positives.
    In a real scenario, an adversary computes hashes to find collisions.
    Here we simulate it by 'mining' negatives until we find FPs.

    Args:
        num_keys: Number of keys
        filter_sim: An actual filter instance to test against.
                    NOTE: If None, this just returns Uniform (fallback).
                    The benchmark runner must pass this in!
    """
    if filter_sim is None:
        # Fallback if no filter provided to mine against
        return generate_uniform(num_keys, seed)

    np.random.seed(seed)

    # 1. Generate Insert Set
    pool_size = num_keys * 10
    all_keys = np.random.choice(range(pool_size * 100), pool_size, replace=False)
    insert_keys = all_keys[:num_keys]

    # Train the simulation filter
    for k in insert_keys:
        filter_sim.add(k)

    # 2. Mine for False Positives
    # We look for keys NOT in insert_keys but that return True in filter_sim
    candidates = all_keys[num_keys:]
    false_positives = []

    for k in candidates:
        if k in filter_sim:
            false_positives.append(k)
        if len(false_positives) >= num_keys * attack_ratio:
            break

    # If we didn't find enough, pad with random negatives
    required = int(num_keys * attack_ratio)
    if len(false_positives) < required:
        # This implies the filter is very good or error rate is very low
        print(
            f"Warning: Only found {len(false_positives)} FPs out of {required} needed."
        )

    # 3. Construct Query Set
    # Mix of FPs (Attack) and Regular Negatives
    # Actually, adversarial usually means purely bad keys to trigger worst cases.
    # Let's return the mined FPs mixed with some random present keys.

    # Query pool: FPs + Some Normal Keys
    # If attack_ratio = 1.0, all queries are FPs.
    fp_array = np.array(false_positives) if false_positives else np.array([], dtype=int)

    # Fill remainder with present keys
    remainder = num_keys - len(fp_array)
    if remainder > 0:
        fillers = insert_keys[:remainder]
        query_keys = np.concatenate([fp_array, fillers])
    else:
        query_keys = fp_array[:num_keys]

    np.random.shuffle(query_keys)

    # True negatives for FPR check: stick to standard random negatives
    true_negatives = all_keys[
        num_keys + len(false_positives) : num_keys * 2 + len(false_positives)
    ]

    return insert_keys, query_keys, true_negatives


def generate_temporal(num_keys=config.NUM_KEYS, seed=config.SEED, phases=3):
    """
    Generates a Temporal Workload.

    Instead of a single list, this returns a list of (insert_keys, query_keys) tuples,
    representing different time phases.

    We simulate "Shifting Hot Set":
    - Phase 1: Key set A is hot.
    - Phase 2: Key set B is hot (overlap with A or disjoint).
    """
    np.random.seed(seed)

    cycle_data = []

    # Base set of all potential keys
    total_universe = num_keys * 2
    all_keys = np.arange(total_universe)
    np.random.shuffle(all_keys)

    # We'll shift the "active" window across the universe
    window_size = num_keys
    step = int(window_size / phases)

    for i in range(phases):
        # Current window of active keys
        start = (i * step) % total_universe
        # Handle wrap around simply by taking modulo indices
        indices = np.arange(start, start + window_size) % total_universe
        active_keys = all_keys[indices]

        # Inserts: In a temporal workload, we might insert new keys that become active.
        # Or we assume they are already there.
        # Let's say we assume a Sliding Window Bloom Filter case:
        # We insert the 'new' keys of this phase.

        # For simplicity in this benchmark:
        # Insert Keys = The active keys for this phase
        # Query Keys = The active keys (Zipfian skew on them)

        # Skew within the active window
        zipf_vals = np.random.zipf(1.2, len(active_keys))
        zipf_indices = (zipf_vals - 1) % len(active_keys)
        query_keys = active_keys[zipf_indices]

        cycle_data.append(
            {"phase": i, "insert_keys": active_keys, "query_keys": query_keys}
        )

    return cycle_data
