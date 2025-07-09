import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)


def get_metrics(y_test, y_pred, prettified=True):
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=1),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1_score": f1_score(y_test, y_pred, zero_division=0),
    }
    if prettified:
        metrics = pd.DataFrame([metrics])
    return metrics


def precision_at_k(y_true, y_proba, k=50):
    y_true, y_proba = np.array(y_true), np.array(y_proba)
    y_pred = np.take(y_true, np.argsort(y_proba)[::-1])[:k]
    return y_pred.sum() / k


def recall_at_k(y_true, y_proba, k=50, zero_division=0):
    y_true, y_proba = np.array(y_true), np.array(y_proba)
    y_pred = np.take(y_true, np.argsort(y_proba)[::-1])[:k]

    if y_true.sum() == 0:
        return zero_division

    return y_pred.sum() / y_true.sum()


def f1_score_at_k(y_true, y_proba, k=50, zero_division=0):
    p_k = precision_at_k(y_true, y_proba, k=k)
    r_k = recall_at_k(y_true, y_proba, k=k, zero_division=zero_division)

    if r_k == zero_division:
        return zero_division

    return 2 * (p_k * r_k) / (p_k + r_k)


def find_failed_selected_cumsum(y_true, y_proba, pos_label=1):
    """Find the number of failed and selected tests after sorting by probability"""
    y_true, y_proba = np.array(y_true), np.array(y_proba)
    sorted_idx = np.argsort(y_proba)[::-1]
    y_true_sorted = np.take(y_true, sorted_idx)

    y_true_sorted = y_true_sorted == pos_label

    failed_tests = np.cumsum(y_true_sorted)
    selected_tests = np.cumsum(np.ones_like(y_true_sorted))

    return failed_tests, selected_tests


# -----------------------------------------------
# Visualization

def plot_confidence(
    lst_failed_tests: list[list[int]],
    lst_selected_tests: list[list[int]],
    labels: list[list[str]],
    figsize=(10, 5),
):
    """Plots confidence curve for each pair of lst_failed_tests[i] and lst_selected_tests[i]"""

    assert (
        len(lst_failed_tests) == len(lst_selected_tests) == len(labels)
    ), "Lengths of lists must be equal"

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    for i in range(len(lst_failed_tests)):
        failed_tests, selected_tests = lst_failed_tests[i], lst_selected_tests[i]

        all_failed, all_tests = max(failed_tests), max(selected_tests)
        failed_percentage, selected_percentage = (
            np.array(failed_tests) / all_failed * 100,
            np.array(selected_tests) / all_tests * 100,
        )
        ax.plot(selected_percentage, failed_percentage, label=labels[i])

        ax.text(0.75, 0.1, f"Number of tests: { all_tests }", transform=ax.transAxes)
        ax.yaxis.set_major_formatter(mtick.PercentFormatter())
        ax.xaxis.set_major_formatter(mtick.PercentFormatter())
        ax.set_xlabel("Selected tests")
        ax.set_ylabel("Confidence")
        ax.set_title("Confidence / selected percentage")
    plt.grid()
    plt.legend()
    plt.show()