## in the script, we want to analyze the 4-gram text between a hypothesis and a reference text
import os
import os.path as osp
import argparse
import numpy as np
from collections import defaultdict

from scipy.stats import entropy
import matplotlib.pyplot as plt

def plot_distributions(p_ref, p_hyps):
    """
    Plot three probability distributions on a single line plot for comparison.

    Args:
    p_a (np.array): Probability distribution A.
    p_b (np.array): Probability distribution B.
    p_c (np.array): Probability distribution C.
    """
    # Create an array for the x-axis that corresponds to the indices of the probabilities
    x = np.arange(len(p_ref))  # Assumption: all distributions have the same number of elements

    # Create a line plot
    plt.figure(figsize=(8, 6))
    plt.plot(x, p_ref, linestyle='-', label='p_a')
    for i, p in enumerate(p_hyps):
        plt.plot(x, p, linestyle='-', label=f'p_{i+1}', alpha=0.5, color=f'C{i}')
    # plt.plot(x, p_b, marker='s', linestyle='--', label='p_b')
    # plt.plot(x, p_c, marker='^', linestyle='-.', label='p_c')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    plt.xlabel('Outcomes')
    plt.ylabel('Probability')
    plt.title('Probability Distributions Comparison')
    # plt.xticks(x, [f'Outcome {i+1}' for i in range(len(p_ref))])
    plt.legend()

    # Display the plot
    # plt.grid(True)
    plt.tight_layout()
    plt.show()
    plt.savefig("distributions.png")

def jensen_shannon_divergence(p, q):
    """
    Calculate the Jensen-Shannon divergence between two probability distributions.

    Args:
    p (np.array): Probability distribution 1.
    q (np.array): Probability distribution 2.

    Returns:
    float: Jensen-Shannon divergence.
    """
    # Normalize to ensure they are proper probability distributions
    p = p / np.sum(p)
    q = q / np.sum(q)
    
    # Calculate the average distribution
    m = 0.5 * (p + q)
    
    # Calculate KL divergences
    kl_p_m = entropy(p, m, base=2)
    kl_q_m = entropy(q, m, base=2)
    
    # Calculate the Jensen-Shannon divergence
    js_divergence = 0.5 * (kl_p_m + kl_q_m)
    
    return js_divergence

def get_ngram_count_dict(fpath, n=4):
    ngram_count_dict = defaultdict(lambda: 0)
    with open(fpath, "r") as f:
        for line in f:
            line = line.strip()
            words = line.split()
            for i in range(len(words) - n + 1):
                ngram = " ".join(words[i:i+n])
                ngram_count_dict[ngram] += 1
    return ngram_count_dict

def main(args):
    print(args)
    hyps = args.hyps.split(",")
    ngram = args.ngram
    hyps_ngram_counts = [get_ngram_count_dict(hyp.strip(), n=ngram) for hyp in hyps]
    ref_ngram_count = get_ngram_count_dict(args.ref, n=ngram)

    ref_ngram_count_list = list(ref_ngram_count.items())
    ref_ngram_count_list.sort(key=lambda x: x[1], reverse=True)
    if args.top_k == 0:
        args.top_k = len(ref_ngram_count_list)
    print(f"Number of 4-gram in ref: {len(ref_ngram_count_list)}, filtered to top {args.top_k}")
    ref_ngram_count_list = ref_ngram_count_list[:args.top_k]
    print(ref_ngram_count_list[:10])
    ref_ngram_count_distribution = np.array([x[1] for x in ref_ngram_count_list])
    ref_ngram_count_distribution = ref_ngram_count_distribution / np.sum(ref_ngram_count_distribution)

    hyps_ngram_count_distributions = []
    for hyp, hyp_ngram_count in zip(hyps, hyps_ngram_counts):
        hyp_ngram_count_list = [hyp_ngram_count[ngram] for ngram, _ in ref_ngram_count_list]
        sorted_hyp_ngram_count_list = sorted(hyp_ngram_count.items(), key=lambda x: x[1], reverse=True)
        print(sorted_hyp_ngram_count_list[:10])

        hyp_ngram_count_distribution = np.array(hyp_ngram_count_list)
        hyp_ngram_count_distribution = hyp_ngram_count_distribution / np.sum(hyp_ngram_count_distribution)
        js_divergence = jensen_shannon_divergence(ref_ngram_count_distribution, hyp_ngram_count_distribution)
        print(js_divergence)
        hyps_ngram_count_distributions.append(hyp_ngram_count_distribution)
    
        js_divergence = jensen_shannon_divergence(ref_ngram_count_distribution, hyp_ngram_count_distribution)

        print(f"JS-divergence of top-{args.top_k} 4-gram in {hyp}: {js_divergence}")
    # plot all distributions
    plot_distributions(ref_ngram_count_distribution, hyps_ngram_count_distributions)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hyps",
        default="",
        help="comma separated hyp file paths",
    )
    parser.add_argument(
        "--ref",
        default="",
        help="ref file path",
    )
    parser.add_argument(
        "--ngram",
        default=4,
        type=int,
        help="ngram to consider",
    
    )
    parser.add_argument(
        "--top_k",
        default=500,
        type=int,
        help="top k ngram to consider",
    )
    args = parser.parse_args()

    main(args)