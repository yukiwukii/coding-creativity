import os
import json
import numpy as np
from scipy.stats import t
import matplotlib.pyplot as plt

def compute_ci_and_mean(results):
    arr = np.array(results)           # shape (n_runs, num_rounds+1)
    means = arr.mean(axis=0)
    stds = arr.std(axis=0, ddof=1)    # sample standard deviation
    n = arr.shape[0]
    tcrit = t.ppf(0.975, n - 1)       # two‐tailed 95% CI
    ci_half = tcrit * (stds / np.sqrt(n))
    return means, ci_half

import matplotlib.pyplot as plt
import numpy as np

def plot_percentage_diff_bar(base_ct, base_dt, base_cr,
                             other_ct, other_dt, other_cr,
                             dp_rounds,
                             method_name="METHOD",
                             out_dir="plots"):
    num_points = dp_rounds + 1
    rounds = np.arange(num_points)

    print("base ct", base_ct)
    print("other ct", other_ct)
    
    ct_diff = [(o - b) for o, b in zip(other_ct, base_ct)]
    dt_diff = [(o - b) for o, b in zip(other_dt, base_dt)]
    cr_diff = [(o - b) for o, b in zip(other_cr, base_cr)]

    print("ct diff", ct_diff)
    print("dt diff", dt_diff)
    print("cr diff", cr_diff)

    
    all_diffs = ct_diff + dt_diff + cr_diff
    y_min, y_max = min(all_diffs), max(all_diffs)
    y_margin = (y_max - y_min) * 0.1  # Add a 10% margin for better visibility
    
    width = 0.25
    fig, ax = plt.subplots()  # Remove figsize to match first script
    
    ax.bar(rounds - width, ct_diff, width, label='Convergent Thinking', color='blue')
    ax.bar(rounds,       dt_diff, width, label='Divergent Thinking',   color='green')
    ax.bar(rounds + width, cr_diff, width, label='Creativity',           color='red')
    
    ax.set_ylim(y_min - y_margin, y_max + y_margin)
    
    ax.axhline(0, color='black', linewidth=1)
    
    ax.set_xlabel('Number of Constraints')
    ax.set_ylabel('Difference in Score (%)')
    ax.set_title(f'Difference in Scores Between {method_name} and BASE Outputs')
    ax.set_xticks(rounds)
    ax.set_xticklabels([str(r) for r in rounds])
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # save
    os.makedirs(out_dir, exist_ok=True)
    fname = os.path.join(out_dir, f"{method_name.lower()}_vs_base.png")
    fig.savefig(fname)
    plt.close(fig)
    print(f"Saved plot to {fname}")

def main():
    with open('datasets/CodeForce/llama8_summary.json') as f:
        data = json.load(f)

    example = next(iter(data.values()))
    dp_rounds = example['dp_rounds']

    grouped = {}
    for rec in data.values():
        t = rec['type']
        grouped.setdefault(t, {'convergent': [], 'divergent': [], 'total': []})
        grouped[t]['convergent'].append(rec['convergent_creativity'])
        grouped[t]['divergent'].append(rec['divergent_creativity'])
        grouped[t]['total'].append(rec['total_creativity'])

    stats = {}
    for t, metrics in grouped.items():
        stats[t] = {}
        for metric_name, runs in metrics.items():
            mean_vals, ci_vals = compute_ci_and_mean(runs)
            stats[t][metric_name] = {
                'mean': mean_vals,
                'ci': ci_vals
            }

    base = stats['base']
    for other_type, other_stats in stats.items():
        if other_type == 'base':
            continue

        plot_percentage_diff_bar(
            base['convergent']['mean'],
            base['divergent']['mean'],
            base['total']['mean'],
            other_stats['convergent']['mean'],
            other_stats['divergent']['mean'],
            other_stats['total']['mean'],
            dp_rounds,
            method_name=other_type
        )

if __name__ == '__main__':
    main()
