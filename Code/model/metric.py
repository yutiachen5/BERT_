# -*- coding: utf-8 -*-

import json
import numpy as np
import pandas as pd
import itertools
from sklearn import metrics
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error


# class MlpMetric(object):

def rmse(y_pred, y_true):
    return mean_squared_error(y_pred=y_pred, y_true=y_true, squared=False)


def mae(y_pred, y_true):
    return mean_absolute_error(y_pred=y_pred, y_true=y_true)


class MAA_metrics(object):
    def __init__(self, token_with_special_list):
        self.token_with_special_list = token_with_special_list

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
        return retval

