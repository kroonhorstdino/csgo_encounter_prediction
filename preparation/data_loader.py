import numpy as np
import pandas as pd

import sys
import json

from pathlib import Path
from typing import Tuple, List

import glob
'''
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset
'''

sys.path.append(Path.cwd().parent)
sys.path.append(Path.cwd())


def get_minibatch_balanced_player(data: pd.DataFrame,
                                  player_i,
                                  batch_size=128,
                                  max_time_to_next_death=5
                                  ) -> Tuple[List[pd.DataFrame], pd.DataFrame]:
    '''
        Returns a minibatch that is balanced for a specific player.

    Returns a minibatch which is a 50/50 (or otherwise specified) split between samples where the selected player is going to die, and won't die in the next x seconds
    '''

    player_dies_mask, player_not_die_mask = get_player_minibatch_mask(
        data, player_i, 5)

    num_sample_from_die = int(batch_size * 0.5)
    num_sample_from_not_die = batch_size - num_sample_from_die

    # Player stays dead
    have_enough_unique_data = sum(player_dies_mask) > num_sample_from_die

    data_batch_die: pd.DataFrame
    try:
        data_batch_die = data[player_dies_mask].sample(
            n=num_sample_from_die, replace=(have_enough_unique_data == False))
    except:
        print("Not enough samples to balance this batch!")
        raise Exception()

    # Player stays alive
    have_enough_unique_data = sum(
        player_not_die_mask) > num_sample_from_not_die
    data_batch_not_die = data[player_not_die_mask].sample(
        n=num_sample_from_not_die, replace=(have_enough_unique_data == False))

    data_batch = pd.concat([data_batch_die, data_batch_not_die])

    return split_data_into_minibatch(data_batch, max_time_to_next_death)


def get_player_minibatch_mask(data: pd.DataFrame,
                              player_i: int,
                              max_time_to_next_death: int = 5,
                              is_binary=True):

    classification_clmn_name = get_die_within_seconds_column_names(
        10)[player_i]
    isAlive_clmn_name = get_isAlive_column_names(10)[player_i]

    isAlive_mask = data[isAlive_clmn_name] == 1
    isDying_mask = data[classification_clmn_name] == 1

    # Instances where player is going to die, meaning being alive and dead in the future
    player_die_mask = isAlive_mask & isDying_mask
    # Simply inverting would leave samples where players are dead and staying dead in the next x seconds
    # Player is alive, and is going to stay alive in the next x seconds
    player_not_die_mask = isAlive_mask & ~isDying_mask

    return player_die_mask, player_not_die_mask


def split_data_into_minibatch(df: pd.DataFrame,
                              max_time_to_next_death: int = 5
                              ) -> Tuple[List[pd.DataFrame], pd.DataFrame]:
    """
    Returns batch with input features and classification labels
    Samples of players are separated in a list
    """

    # Get classification labels for players
    classification_column_names = get_die_within_seconds_column_names(
        10, max_time_to_next_death)
    classification_labels = df[classification_column_names].to_numpy()

    player_features = get_all_player_features_array(df)

    return player_features, classification_labels


def get_all_player_features_array(df: pd.DataFrame) -> List[pd.DataFrame]:
    '''
    Separate player features into Dataframes from each player as a list
    '''
    player_features = []

    for player_i in range(10):
        # Filter feature columns for each player, without classification labels
        player_features.append(df.filter(like=f'f_{player_i}_').to_numpy())

    return player_features


def get_isAlive_column_names(num_players: int = 10) -> List[str]:

    column_names = []

    for player_i in range(num_players):
        column_names.append(f'f_{player_i}_IsAlive')

    return column_names


def get_die_within_seconds_column_names(num_players=10,
                                        time_window_to_next_death=5
                                        ) -> List[str]:
    """
    Gets exact names for deathState classification columns
    """
    actual_column_names = []

    for index in range(num_players):
        actual_column_name = 'l_' + str(index) + '_die_within_' + \
            str(time_window_to_next_death) + '_seconds'
        actual_column_names.append(actual_column_name)

    return actual_column_names


def get_num_player_features(column_labels: List[str]):
    def filter_features(feature_label: str):
        return feature_label.startswith('f_')

    return len(list(filter(filter_features, column_labels.values)))


def get_files_in_directory(files_path: Path,
                           file_extension: str) -> List[Path]:
    '''
    Get all files in specified directory with matching extension
    '''
    glob_ext = f'*{file_extension}'
    files_list = list(Path(files_path).glob(glob_ext))
    return list(files_list)


def load_csv_as_df(filePath: Path) -> pd.DataFrame:
    '''
    Loads parsed .csv file of match as df with all nessecary index modifications and NaN handling
    Removes Round column and sets Tick column as index for dataframe
    '''

    df = pd.read_csv(filePath, sep=',', na_values='-').astype(np.float32)

    df.set_index('Tick', inplace=True)
    # NOTE: Don't drop if still relevant
    df.drop(columns=['Round'], inplace=True)
    df.fillna(0.0, inplace=True)

    return df


def load_h5_as_df(filePath: Path, drop_ticks: bool) -> pd.DataFrame:
    '''
    Loads .h5 file as dataframe

    If true drops 'Tick' as index and removes that column it from df
    '''

    df = pd.read_hdf(filePath).astype(np.float32)

    if ('Tick' in df.columns and drop_ticks):
        # Reset index and drop it
        df.reset_index(drop=True, inplace=True)

    df.fillna(0.0, inplace=True)

    return df


def load_config(config_path: Path):
    config = None
    try:
        with open(str(config_path)) as json_file:
            print("Loading config from " + config_path)
            config = json.load(json_file)
    except:
        print("Couldn't load config file, using default one")
        with open('config/dataset_config.json') as json_file:
            config = json.load(json_file)

    return config

    # Testing
if __name__ == "__main__":
    '''
    features, labels= get_minibatch_balanced_player(pd.read_csv(
        Path.cwd() / 'parsed_files' / 'positions.csv', sep=',', na_values='-'), 0)

    print(features)
    print(labels)
    '''
