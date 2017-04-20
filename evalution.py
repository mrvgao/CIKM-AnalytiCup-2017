'''
Evalution the performance of Regression Result.
Input is the Prediction results
    [yhat_0, yhat_1, ..., yhat_n]
    [y_0, y_1, ...., y_n]
'''

import tensorflow as tf
import numpy as np


def RMSE(yhat, y):
    # RMSE
    return np.sqrt(np.mean((yhat - y) ** 2))


yhat = np.array([1, 2, 3])
y = np.array([1, 0, 3])
L = RMSE(yhat, y)

assert L != 0
assert L is not None
assert isinstance(L, float)



