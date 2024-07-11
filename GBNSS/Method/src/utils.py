"""
Data processing utilities
"""

import os
import math
from texttable import Texttable
import numpy as np

def tab_printer(args):
    """
    Function to print the logs in a nice tabular format.
    :param args: Parameters used for the model.
    """
    args = vars(args)
    keys = sorted(args.keys())
    t = Texttable()
    t.add_rows([["Parameter", "Value"]])
    t.add_rows([[k.replace("_", " ").capitalize(), args[k]] for k in keys])
    print(t.draw())



def calculate_loss(prediction, target):
    """
    Calculating the squared loss on the normalized gosim.
    :param prediction: Predicted log value of gosim.
    :param target: Factual log value of gosim.
    :return score: Squared error.
    """
    prediction = -math.log(prediction)
    target = -math.log(target)
    score = (prediction - target)**2
    return score

def calculate_normalized_gosim(data):
    """
    Calculate the normalized gosim for a pair of graphs.
    :param dta: Data table.
    :return norm_gosim: Normalized gosim score.
    """
    norm_gosim = data["target"] / (0.5*(len(data["features_1"]) + len(data["features_2"])))
    return norm_gosim

def train_test_split(data, ratio=0.7):
    n_samples = len(data)
    n_train = int(n_samples * ratio)
    np.random.shuffle(data)
    train_data = data[:n_train]
    test_data = data[n_train:]
    return train_data, test_data