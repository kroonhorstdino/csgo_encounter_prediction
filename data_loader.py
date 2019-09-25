import numpy as np
import pandas as pd

from pathlib import Path

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset


def get_minibatch_simple(data):
    """ Returns minibatch with input features and classification labels"""

    classification_column_names = get_isAlive_column_names(10)

    # Get classification labels for players
    classification_labels = data[classification_column_names]
    # Drop classification labels and return the rest
    player_features = data.drop(columns=classification_column_names)

    return player_features, classification_labels


def get_isAlive_column_names(num_players):
    """Gets exact names for isAlive columns
        e.g. ['0_isAlive','1_isAlive'....]
    """

    return get_player_column_names(num_players, "_isAlive")


def get_lifeState_column_names(num_players, max_seconds_in_future):
    """Gets exact names for lifeState classification columns
        e.g. ['0_isAlive','1_isAlive'....]
    """

    # TODO Labeling
    lifeState_column_name = "_lifeState_within_" + max_seconds_in_future + "_seconds"

    return get_player_column_names(
        num_players, lifeState_column_name)


def get_player_column_names(num_players, general_column_name):
    actual_column_names = []

    for index in range(num_players):
        general_column_name = str(index) + general_column_name
        actual_column_names.append(general_column_name)

    return actual_column_names


if __name__ == "__main__":
    x = get_isAlive_column_names(10)
    features, labels = get_minibatch_simple(pd.read_csv(
        Path.cwd() / 'parsed_files' / 'positions.csv', sep=',', na_values='-'))

    print(features)
    print(labels)
