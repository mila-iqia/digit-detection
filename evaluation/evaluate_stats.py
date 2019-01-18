import argparse
from pathlib import Path
import numpy as np
from scipy.stats import ttest_ind


def evaluate_stats(y_true, y_pred, y_benchmark):

    assert len(y_true) == len(y_pred), "Amount of samples is not identical"

    acc_benchmark = (y_benchmark == y_true)
    acc_pred = (y_pred == y_true)

    assert len(acc_pred.shape) == 1
    assert len(acc_benchmark.shape) == 1

    stat, p_value = ttest_ind(acc_benchmark, acc_pred, equal_var=False)

    print("\nP-value score for the t-test : ", p_value)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--results_dir", type=str, default='')
    # results_dir will be an absolute path to a directory where the output of
    # your inference will be saved.

    args = parser.parse_args()
    results_dir = args.results_dir

    results_dir = Path(results_dir)
    y_pred = np.loadtxt(results_dir / 'eval_pred.txt')
    y_true = np.loadtxt(results_dir / 'eval_true.txt')
    y_benchmark = np.loadtxt(results_dir / 'eval_benchmark.txt')

    evaluate_stats(y_true, y_pred, y_benchmark)
