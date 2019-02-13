import argparse
import numpy as np
from scipy.stats import wilcoxon


def evaluate_stats(y_true, y_pred1, y_pred2):
    '''
    Statistical evaluation. Compare the distribution of the prediction
    of two different models using the Wilcoxon signed-rank test and
    print the p-value.

    Parameters
    ----------
    y_true: ndarray
        The ground truth labels.
    y_pred1: ndarray
        The prediction of the first model.
    y_pred2: ndarray
        The prediction of the second model.

    '''

    assert len(y_true) == len(y_pred1), "# of samples differ"
    assert len(y_true) == len(y_pred2), "# of samples differ"

    acc_pred1 = (y_pred1 == y_true) * 1
    acc_pred2 = (y_pred2 == y_true) * 1

    def acc(y):
        return np.sum(y) / len(y)
    print("Accuracy of first model", acc(acc_pred1))
    print("Accuracy of second model", acc(acc_pred2))

    stat, p_value = wilcoxon(acc_pred1, acc_pred2, zero_method='zsplit')
    # One-sided p_value
    p_value = p_value / 2

    print('''\nP-value score for the
          Wilcoxon signed-rank test : {:.4f}'''.format(p_value))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--y_true", type=str,
                        default='/results/ground_truth.txt',
                        help='''y_true is the absolute path to the
                                file where the ground truth is
                                saved.''')

    parser.add_argument("--y_pred1", type=str,
                        default='/results/eval_pred1.txt',
                        help='''y_pred1 is the absolute path to the
                                file where the output of model 1
                                inference is saved.''')

    parser.add_argument("--y_pred2", type=str,
                        default='/results/eval_pred2.txt',
                        help='''y_pred2 is the absolute path to the
                                file where the output of model 2
                                inference is saved.''')

    args = parser.parse_args()
    y_true = np.loadtxt(args.y_true)
    y_pred1 = np.loadtxt(args.y_pred1)
    y_pred2 = np.loadtxt(args.y_pred2)

    evaluate_stats(y_true, y_pred1, y_pred2)
