# -*- coding: utf-8 -*-

import json
import numpy as np
import pandas as pd
import itertools
from sklearn import metrics


def accuracy_sample(y_pred, y_true):
    """Compute the accuracy for each sample

    Note that the input y_pred and y_true are both numpy array.
    Args:
        y_pred (numpy.array): shape [seq_len*2, batch_size, ntoken]
        y_true (numpy.array): shape [seq_len*2, batch_size]
    """
    y_pred = y_pred.argmax(axis=2)
    print('Shape of y_pred:', y_pred.shape)
    print('Shape of y_true:', y_true.shape)
    return metrics.accuracy_score(y_pred=y_pred, y_true=y_true)


def accuracy_amino_acid(y_pred, y_true):
    '''Compute teh accuracy for each amino acid.
    Args:
        y_pred (numpy.array): shape [batch_size, seq_len, ntoken]
        y_true (numpy.array): shape [batch_size, seq_len]
    '''
    y_pred = y_pred.argmax(axis=2)
    return metrics.accuracy_score(y_pred=y_pred.flatten(), y_true=y_true.flatten())


def correct_count_seq(y_pred, y_true):
    '''Count the correct prediction for each amino acid.

    Args:
        y_pred (numpy.array): shape [batch_size, seq_len, ntoken]
        y_true (numpy.array): shape [batch_size, seq_len]
    '''
    y_pred = y_pred.argmax(axis=2)
    y_true_copy = y_true.copy()
    y_true_copy[y_true_copy==21] = -100
    return (y_pred == y_true_copy).sum(), np.count_nonzero(y_true_copy != -100)


class metrics(object):
    def __init__(self, token_with_special_list):
        self.token_with_special_list = token_with_special_list
        self.AMINO_ACIDS = "ACTNG"

    def compute_metrics(self, pred, top_n=3):
        """
        Compute metrics to report
        top_n controls the top_n accuracy reported
        """
        # labels are -100 for masked tokens and value to predict for masked token
        labels = pred.label_ids.squeeze()  # Shape (n, 32)
        preds = pred.predictions  # Shape (n, 32, 26)

        n_mask_total = 0
        top_one_correct, top_n_correct = 0, 0
        blosum_values = []
        for i in range(labels.shape[0]):
            masked_idx = np.where(labels[i] != -100)[0]
            n_mask = len(masked_idx)  # Number of masked items
            n_mask_total += n_mask
            pred_arr = preds[i, masked_idx]
            truth = labels[i, masked_idx]  # The masked token indices
            # argsort returns indices in ASCENDING order
            pred_sort_idx = np.argsort(pred_arr, axis=1)  # apply along vocab axis
            # Increments by number of correct in top 1
            top_one_correct += np.sum(truth == pred_sort_idx[:, -1])
            top_n_preds = pred_sort_idx[:, -top_n:]
            for truth_idx, top_n_idx in zip(truth, top_n_preds):
                # Increment top n accuracy
                top_n_correct += truth_idx in top_n_idx

        # These should not exceed each other
        assert top_one_correct <= top_n_correct <= n_mask_total

        retval = {
            f"top_{top_n}_acc": top_n_correct / n_mask_total,
            "acc": top_one_correct / n_mask_total}
        print(retval)

        return retval


def accuracy(y_pred, y_true):
    """
    Note that the input y_pred and y_true are both numpy array.
    Args:
        y_pred (numpy.array): 
        y_true (numpy.array): [description]
    """
    return metrics.accuracy_score(y_pred=y_pred.round(), y_true=y_true)


def recall(y_pred, y_true):
    """
    Note that the input y_pred and y_true are both numpy array.
    Args:
        y_pred ([type]): [description]
        y_true ([type]): [description]
    Returns:
        [type]: [description]
    """
    return metrics.recall_score(y_pred=y_pred.round(), y_true=y_true)


def roc_auc(y_pred, y_true):
    """
    The values of y_pred can be decimicals, within 0 and 1.
    Args:
        y_pred ([type]): [description]
        y_true ([type]): [description]
    Returns:
        [type]: [description]
    """
    return metrics.roc_auc_score(y_score=y_pred, y_true=y_true)
