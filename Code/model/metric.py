# -*- coding: utf-8 -*-

import json
import numpy as np
import pandas as pd
import itertools
from sklearn import metrics
from sklearn.metrics import mean_squared_error


class metrics(object):

    def rmse_cal(y_pred, y_true):
        return mean_squared_error(y_pred=y_pred, y_true=y_true, squared=False)


