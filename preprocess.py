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

def add_death_in_seconds_labels(df, max_time_to_next_death=5, demo_tickrate=128, tickrate_per_second=8):
    """
    Adds labels that contain time to next death within x next seconds to the dataframe

    Assumes tick rate is 8 per second for all dataframes
    """

    print("Adding life state labels for " +
          str(max_time_to_next_death) + " seconds to dataframe...")

    # How many rows in the future have to be considered for labeling
    max_rows_in_future = tickrate_per_second * max_time_to_next_death
    # Max time at which to label if someone is going to die in the next x seconds
    max_ticks_in_future = demo_tickrate * max_time_to_next_death

    isAlive_colum_names = data_loader.get_isAlive_column_names(10)
    isAlive_columns = df[isAlive_colum_names]

    # Contains the columns that hold labels for death within x next seconds
    # 0: not dead 1: is dead
    label_deathState_column_lists = [[] for i in range(10)]

    # Go through each player and set deathState labels for entire column
    # TODO Only set deathState for when player is alive. For performance reason?. Maybe unnecessary
    for player_i, player_isAlive_column_name in enumerate(isAlive_colum_names):
        # Get isAlive column for player
        isAlive_column = isAlive_columns[[player_isAlive_column_name]]
        isAlive_column = isAlive_column[isAlive_column[player_isAlive_column_name] == 0]

        # Remove rows if they are not far enough in the future, meaning they are less than x seconds in the future from the first tick of the first round
        # TODO Ticks begin at warmup, so may never be used
        if df.index[0] > (isAlive_column.index[0] - max_ticks_in_future):
            isAlive_column = isAlive_column[max_rows_in_future:]
            # Only get rows where a player is dead
        label_deathState_column_list = np.full(df.index.size, 0)

        # Go through all rows of this player. Will not set states at end of rounds or at end of segments with discarded ticks, because not future data is available
        # TODO Use death times in the future
        for currentTick, deathState_row in isAlive_column.iterrows():  # TODO deathState_row wird nicht benutzt
            past_tick = (currentTick - max_ticks_in_future)

            if past_tick in df.index: #Past tick may have been discarded during parsing
				label_deathState_column_list[df.index.get_loc(past_tick)] = 1  # Player is going to die in the next x seconds at this tick

        label_deathState_column_lists[player_i] = label_deathState_column_list

    # Add deathState lists into df as columns for each player
    for player_i, label_deathState_column_list in enumerate(label_deathState_column_lists):
        # TODO
        new_column_name = f'l_{player_i}_DeathState_in_{max_time_to_next_death}_seconds'

        df[new_column_name] = label_deathState_column_list.astype(np.float32)

    # print(df.head(20))

    return df

def sample_data(df):
	pass

def normalize(df):
    pass


def randomize_files(fileList,num_of_chunks=10,max_num_of_files=None):

	if max_num_of_files == None:
		max_num_of_files = len(fileList)

	#size_of_files = sum(os.path.getsize(f) for f in os.listdir('.') if os.path.isfile(f))
	df = load_file_as_df(fileList[0])

	if max_num_of_files > 0:
		for file in fileList[1:max_num_of_files]
			df = pd.concat(df, load_file_as_df(file))

	df = df.sample(frac=1) #Shuffling #TODO maybe sklearn shuffle?

	df.to_hdf(str(Path.cwd() // 'parsed_files' // 'data.h5'), key='df', mode='w') #TODO Split into chunks!

def load_file_as_df(filePath):
	df = pd.read_csv(filePath, sep=',', na_values='-').astype(np.float32)

	df.set_index('Tick', inplace=True)
	# TODO Don't drop if still relevant
	df.drop(columns=['Round'], inplace=True)
	df.fillna(0.0, inplace=True)

	return df


if __name__ == "__main__":
    x = add_death_in_seconds_labels(pd.read_csv(
        Path.cwd() / 'parsed_files' / 'positions.csv', sep=',', na_values='-'))

    print(x)
