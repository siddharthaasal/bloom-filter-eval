# Bloom Filter Evaluation Benchmarks

This project empirically evaluates different probabilistic filter implementations under skewed workloads.

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running Benchmarks

Execute the main benchmark script:

```bash
python benchmarks.py
```

Results are saved in `results/csv/`.

## Configuration

Modify `config.py` to adjust:

- Number of keys
- Number of trials
- Zipfian skew parameters
