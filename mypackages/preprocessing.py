import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from scipy.optimize import linear_sum_assignment as linear_assignment


def accuracy(true_row_labels, predicted_row_labels):
    """Get the best accuracy.

    Parameters
    ----------
    true_row_labels: array-like
        The true row labels, given as external information
    predicted_row_labels: array-like
        The row labels predicted by the model

    Returns
    -------
    float
        Best value of accuracy
    """

    cm = confusion_matrix(true_row_labels, predicted_row_labels)
    indexes = linear_assignment(_make_cost_m(cm))
    total = 0
    for row, column in zip(indexes[0], indexes[1]):
        value = cm[row][column]
        total += value

    return (total * 1. / np.sum(cm))



def _make_cost_m(cm):
    s = np.max(cm)
    return (- cm + s)

def min_max_scaler(df):
    for i in range(len(df.columns)-1):
        df[i] = (df[i]-df[i].min())/(df[i].max()-df[i].min())
        
    return df