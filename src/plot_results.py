import pandas as pd
import matplotlib.pyplot as plt
import ast
import os
import seaborn as sns

# Set style for premium/publication-quality plots
plt.style.use('seaborn-v0_8-paper')
sns.set_theme(style="whitegrid")
sns.set_palette("colorblind")

RESULTS_DIR = "results"
CSV_PATH = os.path.join(RESULTS_DIR, "csv", "research_benchmark_results.csv")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")

def load_and_preprocess_data(csv_path):
    """Loads CSV and extracts parameters from the string representation."""
    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found at {csv_path}")
        return None

    df = pd.read_csv(csv_path)
    
    # Parse the dictionary string in 'Params' column
    def safe_parse(val):
        try:
            return ast.literal_eval(val)
        except (ValueError, SyntaxError):
            return {}

    df['Params_Dict'] = df['Params'].apply(safe_parse)
    
    # Extract specific parameters into new columns
    df['Alpha'] = df['Params_Dict'].apply(lambda x: x.get('alpha', None))
    df['AttackRatio'] = df['Params_Dict'].apply(lambda x: x.get('attack_ratio', None))
    
    return df

def plot_baseline_comparison(df):
    """Generates bar charts for Uniform Workload (Baseline)."""
    print("Generating baseline comparisons...")
    uniform_df = df[df['Workload'] == 'generate_uniform'].copy()
    
    if uniform_df.empty:
        print("No uniform workload data found.")
        return

    # Metrics to plot
    metrics = [
        ('FPR', 'False Positive Rate', 'Lower is Better'),
        ('BitsPerKey', 'Memory Usage (Bits per Key)', 'Lower is Better'),
        ('InsertLatency(us)', 'Insert Latency (µs)', 'Lower is Better'),
        ('QueryLatency(us)', 'Query Latency (µs)', 'Lower is Better')
    ]

    for col, title, ylabel in metrics:
        plt.figure(figsize=(8, 5))
        sns.barplot(data=uniform_df, x='Filter', y=col, errorbar='sd', capsize=.1)
        plt.title(f'Baseline Performance: {title}', fontsize=12, fontweight='bold')
        plt.ylabel(ylabel)
        plt.xlabel('Filter Type')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, f'baseline_{col.lower().split("(")[0]}.png'))
        plt.close()

def plot_skew_impact(df):
    """Generates line charts showing impact of Zipfian skew."""
    print("Generating skew impact plots...")
    zipf_df = df[df['Workload'] == 'generate_zipfian'].copy()
    
    if zipf_df.empty:
        print("No zipfian workload data found.")
        return

    # Ensure Alpha is numeric for plotting
    zipf_df['Alpha'] = pd.to_numeric(zipf_df['Alpha'])

    metrics = [
        ('FPR', 'False Positive Rate (vs Skew)', 'FPR'),
        ('QueryLatency(us)', 'Query Latency (vs Skew)', 'Latency (µs)')
    ]

    for col, title, ylabel in metrics:
        plt.figure(figsize=(10, 5))
        sns.lineplot(data=zipf_df, x='Alpha', y=col, hue='Filter', style='Filter', markers=True, dashes=False)
        plt.title(f'Impact of Data Skew: {title}', fontsize=12, fontweight='bold')
        plt.xlabel('Zipfian Parameter (Alpha)')
        plt.ylabel(ylabel)
        if 'FPR' in col:
            plt.yscale('log')
            
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title='Filter', frameon=True)
        plt.grid(True, linestyle='--', alpha=0.7, which='both')
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, f'skew_{col.lower().split("(")[0]}.png'))
        plt.close()

def plot_adversarial_impact(df):
    """Compares FPR of normal uniform vs adversarial workloads."""
    print("Generating adversarial impact plots...")
    
    # Filter for Adversarial and Uniform (Baseline)
    adv_df = df[df['Workload'] == 'generate_adversarial'].copy()
    uni_df = df[df['Workload'] == 'generate_uniform'].copy()
    
    if adv_df.empty:
        print("No adversarial data found.")
        return

    # Create a comparative DataFrame
    uni_df['Condition'] = 'Baseline (Uniform)'
    adv_df['Condition'] = 'Adversarial Attack'
    
    combined_df = pd.concat([uni_df, adv_df])

    plt.figure(figsize=(9, 6))
    sns.barplot(data=combined_df, x='Filter', y='FPR', hue='Condition')
    plt.title('Adversarial Resilience: False Positive Rate', fontsize=12, fontweight='bold')
    plt.ylabel('False Positive Rate')
    plt.yscale('log') # Log scale is often better for FPR differences if they are large
    plt.grid(axis='y', linestyle='--', alpha=0.7, which='both')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'adversarial_fpr_impact.png'))
    plt.close()

def plot_adversarial_latency_comparison(df):
    """Generates grouped bar chart for P99 Query Latency: Uniform vs Adversarial."""
    print("Generating adversarial latency plots...")
    
    # Filter for Adversarial and Uniform
    adv_df = df[df['Workload'] == 'generate_adversarial'].copy()
    uni_df = df[df['Workload'] == 'generate_uniform'].copy()
    
    if adv_df.empty or uni_df.empty:
        print("Insufficient data for adversarial latency comparison.")
        return

    # Create a comparative DataFrame
    uni_df['Condition'] = 'Uniform'
    adv_df['Condition'] = 'Adversarial'
    
    combined_df = pd.concat([uni_df, adv_df])

    plt.figure(figsize=(9, 6))
    sns.barplot(data=combined_df, x='Filter', y='P99_QueryLatency(us)', hue='Condition')
    plt.title('P99 Query Latency: Uniform vs Adversarial Workload', fontsize=12, fontweight='bold')
    plt.ylabel('P99 Query Latency (µs)')
    plt.yscale('log')
    plt.grid(axis='y', linestyle='--', alpha=0.7, which='both')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'adversarial_p99_latency.png'))
    plt.close()

def plot_skew_p99_latency(df):
    """Generates line chart for Impact of Data Skew: P99 Query Latency."""
    print("Generating skew P99 latency plots...")
    zipf_df = df[df['Workload'] == 'generate_zipfian'].copy()
    
    if zipf_df.empty:
        print("No zipfian workload data found.")
        return

    # Ensure Alpha is numeric
    zipf_df['Alpha'] = pd.to_numeric(zipf_df['Alpha'])

    plt.figure(figsize=(10, 5))
    sns.lineplot(data=zipf_df, x='Alpha', y='P99_QueryLatency(us)', hue='Filter', style='Filter', markers=True, dashes=False)
    plt.title('Impact of Data Skew: P99 Query Latency', fontsize=12, fontweight='bold')
    plt.xlabel('Zipfian Parameter (Alpha)')
    plt.ylabel('P99 Query Latency (µs)')
    plt.yscale('log')
    
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title='Filter', frameon=True)
    plt.grid(True, linestyle='--', alpha=0.7, which='both')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'skew_p99_latency.png'))
    plt.close()

def generate_plots(csv_path=CSV_PATH, output_dir=PLOTS_DIR):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    df = load_and_preprocess_data(csv_path)
    
    if df is not None:
        plot_baseline_comparison(df)
        plot_skew_impact(df)
        plot_adversarial_impact(df)
        plot_adversarial_latency_comparison(df)
        plot_skew_p99_latency(df)
        print(f"\nAll plots generated in: {os.path.abspath(output_dir)}")

def main():
    generate_plots()

if __name__ == "__main__":
    main()
