# -*- coding: utf-8 -*-

import json
import numpy as np
import pandas as pd
import itertools
from sklearn import metrics
from sklearn.metrics import mean_squared_error


class MLPmetrics(object):

    def rmse_cal(self, pred):
        true_val = pred.label_ids.squeeze()  # Returns a tensor with all specified dimensions of size 1 removed.
        pred_val = pred.predictions
        eva = {'rmse': mean_squared_error(y_pred=true_val, y_true=pred_val, squared=False)}
        return eva


