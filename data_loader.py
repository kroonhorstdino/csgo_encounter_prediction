import numpy as np
import pandas as pd

from pathlib import Path

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset


def split_data_into_minibatch(data,max_time_to_next_death=5):
    """ 
    Returns batch with input features and classification labels
    Samples of players are separated into arrays
    """

    # Get classification labels for players
    classification_column_names = get_deathState_column_names(10, max_time_to_next_death)
    classification_labels = data[classification_column_names]

	player_features = []
	for player_i in range(num_players):
		# Filter feature columns for each player, without classification labels
		player_features.append( df.filter(like=f'f_{player_i}_').to_numpy() )

    return player_features, classification_labels
	

def get_minibatch_balanced_player(data,player_i,batch_size=128,max_time_to_next_death=5):
	'''
	Returns a minibatch that is balanced for a specific player.
	minibatch is a 50/50 split between samples where the selected player is going to die, and won't die in the next x seconds
	'''

	#Get columns belonging to the player
	classification_column_name = get_deathState_column_names(10, max_time_to_next_death)[player_i]
	isAlive_column_name = get_isAlive_column_names(10)[player_i]

	#Get a mask of all samples where the player is going to die, and is also still alive(!)
    player_dies_mask = data[isAlive_column_name] == 1 & data[classification_column_name] == 1
	#Simply inverting would leave samples where players are dead and staying dead in the next x seconds
	player_not_die_mask = data[isAlive_column_name] == 1 & data[classification_column_name] == 0

    num_sample_from_die = int(batch_size/2)
    num_sample_from_not_die = batch_size - num_sample_from_die

	#Player stays dead
    have_enough_unique_data = sum(player_dies_mask) > num_sample_from_die 
    data_batch_die = data[player_dies_mask].sample(n=num_sample_from_die,replace=(have_enough_unique_data == False))

	#Player stays alive
    have_enough_unique_data = sum(player_not_die_mask) > num_sample_from_not_die 
    data_batch_not_die = data[player_not_die_mask].sample(n=num_sample_from_not_die,replace=(have_enough_unique_data == False))

    data_batch = pd.concat([data_batch_die, data_batch_not_die])

	return split_data_into_minibatch(data_batch,max_time_to_next_death)

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

#Testing
if __name__ == "__main__":
    features, labels = get_minibatch_balanced_player(pd.read_csv(
        Path.cwd() / 'parsed_files' / 'positions.csv', sep=',', na_values='-'),0)

    print(features)
    print(labels)
