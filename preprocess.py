import numpy as np
import pandas as pd

from pathlib import Path

import data_loader

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset


def preprocess(filePath):
    pass


def add_death_in_seconds_labels(df, max_time_to_next_death=5):
    """
    Adds labels that contain time to next death within x next seconds to the dataframe

    Assumes tick rate is 8 per second for all dataframes

    Moved into parsing
    """

    # TODO Tickrate
    tickrate_per_second = 8
    max_ticks_in_future = tickrate_per_second * max_time_to_next_death

    isAlive_colum_names = data_loader.get_isAlive_column_names(10)

    column_names_with_ticks = ["Tick"]
    column_names_with_ticks.extend(isAlive_colum_names)

    # Get all rows about life state of players including ticks
    isAlive_columns = df[column_names_with_ticks]
    last_row = isAlive_columns.iloc[0]

    # Contains the columns that hold labels for death within x next seconds
    # 0: not dead 1: is dead
    deathState_column_lists = []

    # Go through each player and set deathState labels for entire column
    for player_i, player_isAlive_column_name in enumerate(isAlive_colum_names):
        # Get isAlive column for player
        isAlive_column = isAlive_columns[["Tick", player_isAlive_column_name]]
        deathState_column_list = np.full(isAlive_columns.index.size, 0.0)

        for index, lifeState_row in isAlive_column.iterrows():
            past_tick = (lifeState_row['Tick'] - max_ticks_in_future)

            if (past_tick >= 0 & & isAlive_columns[index]):
                deathState_column_list[index] = row[player_isAlive_column_name][0]

    deathState_column_lists[player_i] = deathState_column_list

    # Add deathState lists into df as columns for each player
    for player_i, deathState_column_list in enumerate(deathState_column_lists):
        # TODO
        new_column_name = f'{player_i}_deathState_in_{max_time_to_next_death}_seconds'

        df[new_column_name] = deathState_column_list

    return df


def normalize(df):
    pass


def randomize(df):
    pass


if __name__ == "__main__":
    x = add_death_in_seconds_labels(pd.read_csv(
        Path.cwd() / 'parsed_files' / 'positions.csv', sep=',', na_values='-'))

    print(x)
