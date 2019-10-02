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


def create_die_within_sec_labels(df, time_window_to_next_death=5, demo_tickrate=128, tickrate_per_second=8) -> pd.DataFrame:
    """
    Adds labels that contain time to next death within x next seconds to the dataframe

    Assumes tick rate is 8 per second for all dataframes
    """

    print("Adding life state labels for " +
      str(time_window_to_next_death) + " seconds to dataframe...")

    # How many rows in the future have to be considered for labeling
    max_rows_in_future = tickrate_per_second * time_window_to_next_death
    # Max time at which to label if someone is going to die in the next x seconds
    max_ticks_in_future = demo_tickrate * time_window_to_next_death

    # Get names of columns
    isAlive_colum_names = data_loader.get_isAlive_column_names(10)
    isAlive_columns = df[isAlive_colum_names]

    # Contains the columns that hold labels for death within x next seconds
    # 0: not dead 1: is dead
    label_deathState_column_lists = [[] for i in range(10)]

    # Go through each player and set die_within labels for entire column
    for player_i, player_isAlive_column_name in enumerate(isAlive_colum_names):
            # Get isAlive column for player
            isAlive_column = isAlive_columns[player_isAlive_column_name]
            # Only choose entries where player is dead
            isAlive_column = isAlive_column[isAlive_column == 0]

            # Fill list with zeroes, because players won't be dead most of the time
            label_deathState_column_list = np.full(df.index.size, 0)

            # Go through all rows of this player.
            # TODO Use death times in the future
            for currentTick, deathState_row in isAlive_column.iterrows():  # TODO deathState_row wird nicht benutzt
                past_tick = (currentTick - max_ticks_in_future)

                if past_tick in df.index:
                    label_deathState_column_list[df.index.get_loc(
                        past_tick)] = 1  # Player is going to die in the next x seconds at this tick

            label_deathState_column_lists[player_i] = label_deathState_column_list

        new_column_names = data_loader.get_die_within_seconds_column_names(
            time_window_to_next_death=time_window_to_next_death)

        # Add deathState lists into df as columns for each player
        for player_i, label_deathState_column_list in enumerate(label_deathState_column_lists):
            # TODO
            # new_column_name = f'l_{player_i}_die_within_in_{max_time_to_next_death}_seconds'

            df[new_column_names[player_i]
            ] = label_deathState_column_list.astype(np.float32)

        # print(df.head(20))

        return df


def normalize(df):
    pass


def randomize(df):
    pass


if __name__ == "__main__":
    x = create_die_within_sec_labels(pd.read_csv(
        Path.cwd() / 'parsed_files' / 'positions.csv', sep=',', na_values='-'))

    print(x)
