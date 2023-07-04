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

