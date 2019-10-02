import numpy as np
import pandas as pd

from pathlib import Path

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset


def get_minibatch_simple(data):
    """ 
    Returns minibatch with input features and classification labels
    Each player has a seperate dataframe
    """

    classification_column_names = get_die_within_seconds_column_names(10, 5)

    # Get classification labels for players
    classification_labels = data[classification_column_names]
    # Drop classification labels and return the rest
    player_features = data.filter(like=f'f_')

    return player_features, classification_labels


def get_isAlive_column_names(num_players=10):

    column_names = []

    for player_i in range(num_players):
        column_names.append(f'f_{player_i}_IsAlive')

    return column_names


def get_die_within_seconds_column_names(num_players=10, time_window_to_next_death=5):
    """
    Gets exact names for deathState classification columns
    """
    actual_column_names = []

    for index in range(num_players):
        actual_column_name = 'l_' + str(index) + '_die_within_' + \
            str(time_window_to_next_death) + '_seconds'
        actual_column_names.append(actual_column_name)

    return actual_column_names


if __name__ == "__main__":
    features, labels = get_minibatch_simple(pd.read_csv(
        Path.cwd() / 'parsed_files' / 'positions.csv', sep=',', na_values='-'))

    print(features)
    print(labels)
