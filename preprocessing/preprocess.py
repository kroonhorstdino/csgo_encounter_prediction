import numpy as np
import pandas as pd

import sys
import os
from pathlib import Path

import data_loader 
import create_dataset

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset

sys.path.append(Path.cwd().parent)
sys.path.append(Path.cwd())


def add_die_within_sec_labels(df: pd.DataFrame, time_window_to_next_death: int = 5, demo_tickrate: int = 128, parsing_tickrate: int = 8) -> pd.DataFrame:
    '''
    Adds labels that contain time to next death within x next seconds to the dataframe

    Assumes tick rate is 8 per second for all dataframes

    IMPORTANT: Data must not be shuffled. This function relies on the ticks being in the correct order!
    '''

    print("Adding 'dies within x seconds' labels for a " +
          str(time_window_to_next_death) + " second time window to dataframe...")

    # How many rows in the future have to be considered for labeling
    max_rows_in_future = parsing_tickrate * time_window_to_next_death
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

        # Go through all rows of this player. Will not set states at end of rounds or at end of segments with discarded ticks, because not future data is available
        # TODO Use death times in the future
        for currentTick, deathState_row in isAlive_column.items():  # TODO deathState_row wird nicht benutzt
            past_tick = (currentTick - max_ticks_in_future)

            if past_tick in df.index:  # Past tick may have been discarded during parsing
                # Player is going to die in the next x seconds at this tick
                index_location = df.index.get_loc(past_tick)
                #debug_test_sample = df.iloc[int(index_location)]

                label_deathState_column_list[index_location] = 1

        label_deathState_column_lists[player_i] = label_deathState_column_list

    new_column_names = data_loader.get_die_within_seconds_column_names(time_window_to_next_death=time_window_to_next_death)

    # Add deathState lists into df as columns for each player
    for player_i, label_deathState_column_list in enumerate(label_deathState_column_lists):
        # TODO
        # new_column_name = f'l_{player_i}_die_within_in_{max_time_to_next_death}_seconds'

        df[new_column_names[player_i]] = label_deathState_column_list.astype(np.float32)

        # print(df.head(20))

    return df


def randomize_processed_files(fileList, size_of_chunks_in_rows=5000, max_num_of_files=None):
    '''
    Puts multiple .h5 together and shuffles them together. After that they are split into roughly equal chunks
    '''

    if max_num_of_files == None:
        max_num_of_files = len(fileList)

    # size_of_files = sum(os.path.getsize(f) for f in os.listdir('.') if os.path.isfile(f))
    df = data_loader.load_file_as_df(fileList[0])

    if max_num_of_files > 0:
        for file in fileList[1:max_num_of_files]:
            df = pd.concat(df, data_loader.load_file_as_df(file))

    df = df.sample(frac=1)# Shuffling #TODO maybe sklearn shuffle?

    df_length = len(df)

    last_chunk_df = None

    #Split dataframe into roughly equal parts (based on row count) and then save them to directory
    for i in range(0, df_length, size_of_chunks_in_rows):
        last_chunk_df = df[i : min(df_length - 1, i + size_of_chunks_in_rows)]

    df.to_hdf(str(Path.cwd() // 'parsed_files' // 'data.h5'),
              key='df', mode='w')  # TODO Split into chunks!


if __name__ == "__main__":
    lst = create_dataset.generate_dataset_file_partitions(Path.cwd() / 'parsed_files')

    randomize_processed_files(lst)
