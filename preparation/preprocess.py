import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset

import sys
import os
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path.cwd() / 'preparation/'))

import data_loader

sys.path.append(Path.cwd().parent)
sys.path.append(Path.cwd())

features_info = data_loader.load_config(
    Path('preparation') / 'features_info.json')


def add_die_within_sec_labels(df: pd.DataFrame,
                              time_window_to_next_death: int = 5,
                              demo_tickrate: int = 64,
                              parsing_tickrate: int = 8) -> pd.DataFrame:
    '''
    Adds labels that contain time to next death within x next seconds to the dataframe

    Assumes tick rate is 8 per second for all dataframes

    IMPORTANT: Data must not be shuffled. This function relies on the ticks being in the correct order!
    '''

    print("Adding 'dies within x seconds' labels for a " +
          str(time_window_to_next_death) +
          " second time window to dataframe...")

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
        isAlive_column_where_dead = isAlive_columns[player_isAlive_column_name]
        # Only choose entries where player is dead
        isAlive_column_where_dead = isAlive_column_where_dead[
            isAlive_column_where_dead == 0]

        # Fill list with zeroes, because players won't be dead most of the time
        label_deathState_column_list = np.full(df.index.size, 0)

        # Go through all rows of this player. Will not set states at end of rounds or at end of segments with discarded ticks,
        # because no further future data is available
        # All rows here are ones where the player is already dead!
        for currentTick, isDead_row in isAlive_column_where_dead.items():
            past_tick = (currentTick - max_ticks_in_future)

            if past_tick in df.index:  # Past tick may have been discarded during parsing
                # Player is going to die in the next x seconds at this tick
                index_location = df.index.get_loc(past_tick)
                # debug_test_sample = df.iloc[int(index_location)]

                #TODO: Test new labelling!
                #If player is already dead in past tick
                if (df[player_isAlive_column_name].loc[past_tick] == 0):
                    label_deathState_column_list[index_location] = 2
                # When player is not yet dead in the past tick
                else:
                    label_deathState_column_list[index_location] = 1

        label_deathState_column_lists[player_i] = label_deathState_column_list

    new_column_names = data_loader.get_die_within_seconds_column_names(
        time_window_to_next_death=time_window_to_next_death)

    # Add deathState lists into df as columns for each player
    for player_i, label_deathState_column_list in enumerate(
            label_deathState_column_lists):
        # TODO:
        # new_column_name = f'l_{player_i}_die_within_in_{max_time_to_next_death}_seconds'

        df[new_column_names[player_i]] = label_deathState_column_list.astype(
            np.float32)

        # print(df.head(20))

    return df


def undersample_pure_not_die_ticks(df: pd.DataFrame,
                                   death_time_window: int = 5,
                                   removal_frac: float = 0.5) -> pd.DataFrame:
    '''
        Remove a certain amount of ticks, which will not experience a death in the next x seconds
    '''
    return df


# TODO: Do one hot encoding for weapons


def one_hot_encoding_weapons(df: pd.DataFrame):
    '''
        Adds one hot encoding for weapons of player
        Removes 'CurrentWeapon' row for each player
    '''

    itemDefinitionIndexMap = features_info["itemDefinitionIndexMap"]
    weapon_one_hot_ids_column_name = []
    actual_column_names = []

    for index in itemDefinitionIndexMap:
        if (int(index) >= 500 or int(index) == 59):  # Indecies of other knifes
            continue

        weapon_one_hot_ids_column_name.append(index)

    weapon_one_hot_ids_column_name.sort(key=int)

    #For each feature name, generate feature name for each player
    for weapon_id in weapon_one_hot_ids_column_name:
        actual_column_names.extend(
            data_loader.get_feature_column_name(f'Weapon_{weapon_id}'))

    weapon_id_df = pd.DataFrame(columns=actual_column_names,
                                index=df.index,
                                dtype=np.float32)

    weapon_id_df.sort_index(axis=1, inplace=True)

    for player_i in range(
            10):  #For each player one hot encode to each weapon id
        player_current_weapon_column_name = data_loader.get_feature_column_name(
            'CurrentWeapon')[
                player_i]  # Name of column with ids of current weapon
        for index, weapon_index in df[
                player_current_weapon_column_name].iteritems():

            if weapon_index == 0:
                weapon_index = 42  #FIXME: Data should not contain any zeroes in CurrentWeapon!!!

            weapon_id_column_name = f'f_{player_i}_Weapon_{int(weapon_index)}'

            #weapon_id_df[weapon_id_column_name].loc[index] = 1  #Weapon is used at this point in time for this player
            weapon_id_df.iat[weapon_id_df.index.get_loc(index),
                             weapon_id_df.columns.get_loc(weapon_id_column_name
                                                          )] = 1

    weapon_id_df.fillna(0, inplace=True)

    current_weapon_columns_names = data_loader.get_feature_column_name(
        'CurrentWeapon')
    df.drop(columns=current_weapon_columns_names, inplace=True)

    df = pd.concat([df, weapon_id_df])

    return df


if __name__ == "__main__":
    one_hot_encoding_weapons(
        data_loader.load_csv_as_df(
            Path('parsed_files/sprout-vs-ex-epsilon-m3-overpass.csv')))
    pass
