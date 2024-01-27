import pandas as pd
import numpy as np
from typing import Union
from scipy.stats import somersd

def gauc(y_true: np.array,
         y_pred: np.array,
         breaks: Union[list, None] = None):
    """
    Function to calculate generalised (ROC-)AUC using Somers' D, through
    the Kendall tau_b implementation.

    The gAUC is a measure of ordinal association between a "true" set of
    observations and the corresponding predicted values.

    Parameters
    ----------
    y_true: np.array
        True values for categories.
    y_pred: np.array
        Model output containing the predicted category.
    breaks: list, int or None
        If None, the inputs are already assumed to categorical,
        otherwise pd.cut will be employed

    Returns
    -------
    dict
        gAUC value
        p-value for estimate
    """
    # initialisation
    y_true = pd.Series(y_true)
    y_pred = pd.Series(y_pred)
    # checks
    assert len(y_true) == len(y_pred), "true and predicted vectors must have same length."

    # prepare categories for
    if isinstance(breaks, list):  # bin if int or list of bins is given
        y_true = pd.cut(y_true, bins=breaks, precision=5, include_lowest=True)
        y_pred = pd.cut(y_pred, bins=breaks, precision=5, include_lowest=True)
    elif breaks is None:  # if None, the inputs are assumed to already be categorical
        y_true = y_true.astype(str)
        y_pred = y_pred.astype(str)
    else:
        raise ValueError("Invalid input for break")

    # Call Somers' D
    res = somersd(x=y_true, y=y_pred)

    return {"statistic": (res.statistic + 1) / 2,
            "pvalue": res.pvalue}

