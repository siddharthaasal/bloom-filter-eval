"""
Configuration parameters for the Bloom Filter Benchmarking project.
"""

# Experiment Settings
NUM_KEYS = 1000  # Number of keys to insert
FILTER_CAPACITY = 100_000  # Designed capacity of the filters
ERROR_RATE = 0.01  # Target false positive rate (1%)
NUM_TRIALS = 5  # Number of runs to average over
SEED = 42  # Random seed for reproducibility

# Zipfian Workload Settings
ZIPF_ALPHAS = [1.0, 1.2, 1.5]  # Skew parameters to test

# Adversarial Settings
ADVERSARIAL_RATIO = 0.5  # Ratio of queries that are chosen False Positives

# Temporal Settings
TEMPORAL_PHASES = 3  # Number of phases in temporal workload

# Paths
RESULTS_DIR = "results"
CSV_DIR = f"{RESULTS_DIR}/csv"
PLOTS_DIR = f"{RESULTS_DIR}/plots"
