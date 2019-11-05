import numpy as np
import pandas as pd
import time
import numbers
import math

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

FEATURES_INFO = data_loader.load_json(Path('config') / 'features_info.json')
#DATASET_CONFIG = data_loader.load_json(Path('config') / 'dataset_config.json')

EYEANGLES_COLUMN_NAMES = data_loader.get_feature_column_names(
    'EyeAnglePitch') + data_loader.get_feature_column_names('EyeAngleYaw')


class vec3():
    def __init__(self, x, y, z):
        (self.x, self.y, self.z) = (x, y, z)

    def __mul__(self, other):
        return vec3(self.x * other, self.y * other, self.z * other)

    def __add__(self, other):
        return vec3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other):
        return vec3(self.x - other.x, self.y - other.y, self.z - other.z)

    def dot(self, other):
        return (self.x * other.x) + (self.y * other.y) + (self.z * other.z)

    def __abs__(self):
        return self.dot(self)

    def norm(self):
        mag = np.sqrt(abs(self))
        return self * (1.0 / np.where(mag == 0, 1, mag))

    def components(self):
        return (self.x, self.y, self.z)

    def extract(self, cond):
        return vec3(extract(cond, self.x), extract(cond, self.y),
                    extract(cond, self.z))

    def place(self, cond):
        r = vec3(np.zeros(cond.shape), np.zeros(cond.shape),
                 np.zeros(cond.shape))
        np.place(r.x, cond, self.x)
        np.place(r.y, cond, self.y)
        np.place(r.z, cond, self.z)
        return r


class Sphere:
    def __init__(self, center, radius, player_i):
        self.c = center
        self.r = radius
        self.player_i = player_i

    def intersects(self, O, D):
        b = 2 * D.dot(O - self.c)
        c = abs(self.c) + abs(O) - 2 * self.c.dot(O) - (self.r * self.r)
        disc = (b**2) - (4 * c)
        sq = np.sqrt(np.maximum(0, disc))
        h0 = (-b - sq) / 2
        h1 = (-b + sq) / 2
        h = np.where((h0 > 0) & (h0 < h1), h0, h1)
        pred = (disc > 0) & (h > 0)
        return pred


def add_die_within_sec_labels(df: pd.DataFrame,
                              time_window_to_next_death: int = 5,
                              demo_tickrate: int = 64,
                              parsing_tickrate: int = 8) -> pd.DataFrame:
    '''
    Adds labels that contain time to next death within x next seconds to the dataframe

    Assumes tick rate is 8 per second for all dataframes

    IMPORTANT: Data must not be shuffled. This function relies on the ticks being in the correct order!
    '''

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
                #FIXME: Use iat insted of loc!
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


def add_one_hot_encoding_weapons(df: pd.DataFrame):
    '''
        Adds one hot encoding for weapons of player. All knifes are mapped to id 42
        Removes 'CurrentWeapon' columns for each player
    '''

    actual_column_names = data_loader.get_one_hot_encoded_weapon_feature_names(
    )

    one_hot_weapon_df = pd.DataFrame(columns=actual_column_names,
                                     index=df.index,
                                     dtype=np.float32)

    one_hot_weapon_df.sort_index(axis=1, inplace=True)

    for player_i in range(
            10):  #For each player one hot encode to each weapon id
        player_current_weapon_column_name = data_loader.get_feature_column_names(
            'CurrentWeapon')[
                player_i]  # Name of column with ids of current weapon
        for index, weapon_index in df[
                player_current_weapon_column_name].iteritems():

            if weapon_index == 0:
                weapon_index = 42  #FIXME: Data should not contain any zeroes in CurrentWeapon!!!

            weapon_id_column_name = f'f_{player_i}_Weapon_{int(weapon_index)}'

            #FIXME: Use iat insted of loc!
            #weapon_id_df[weapon_id_column_name].loc[index] = 1  #Weapon is used at this point in time for this player
            one_hot_weapon_df.at[index, weapon_id_column_name] = 1

    one_hot_weapon_df.fillna(0, inplace=True)

    current_weapon_columns_names = data_loader.get_feature_column_names(
        'CurrentWeapon')
    df.drop(columns=current_weapon_columns_names, inplace=True)

    df = pd.concat([df, one_hot_weapon_df], sort=True)

    return df


#TODO: Add one hot angles
def add_one_hot_encoding_angles(df: pd.DataFrame, discrete=True):
    '''
        Adds one hot encoding for angle to player (is a raycast within radius of enemy player)
        Removes 'EyeAnglesPitch' and 'EyeAnglesYaw' columns for each player
    '''

    #STOLEN FROM https://github.com/jamesbowman/raytrace/blob/master/rt3.py

    #Iterate through rows
    for index_label in df.index:
        player_positions_vec3 = all_player_position_to_vec3(df, index_label)
        player_looking_directions_vec3 = all_player_eyeAngles_to_direction_vec3(
            df, index_label)

        player_spheres = np.array(
            generate_player_spheres(player_positions_vec3, 5))

        #Iterate through all players and their positions and angles
        for player_i, player_position, player_looking_direction in zip(
                range(10), player_positions_vec3,
                player_looking_directions_vec3):

            #Get all intersections of ray shot from player with other spheres
            get_player_aim_on_enemy(player_position, player_looking_direction,
                                    np.delete(player_spheres, player_i))

            #iterate through enemies
            for enemy_i in range(5):

                #Indexing for list is reversed due to generating of multiple feature column name lists
                df.at[index_label, AIM_ON_ENEMY_COLUMN_NAMES[enemy_i]
                      [player_i]]
            #TODO: Finish calculation of angles

    df.drop(columns=EYEANGLES_COLUMN_NAMES, inplace=True)
    return df


def get_player_aim_on_enemy(player_position: vec3, player_direction: vec3,
                            target_spheres: List[Sphere]):
    # O is the ray origin, D is the normalized ray direction
    # player_spheres are the custom sphere objects of players
    # Returns one hot encoded array for 1 aim on enemy or 0 no aim on enemy

    target_spheres.sort(lambda sph: sph.player_i)

    player_raycast_hits = [
        target_sphere.intersects(player_position, player_direction)
        for target_sphere in target_spheres
    ]

    return player_raycast_hits


def generate_player_spheres(player_positions: List[vec3], radius: float):
    player_spheres = []
    #TODO: Do player spheres

    return player_spheres


def all_player_position_to_vec3(df: pd.DataFrame, index_label) -> List[vec3]:
    all_player_position_as_vec3_list = []

    row = df.loc[index_label]  # Row as series

    for player_i in range(10):
        #Add new vector
        all_player_position_as_vec3_list.append(
            vec3(row[VELOCITYX_COLUMN_NAMES[player_i]],
                 row[VELOCITYY_COLUMN_NAMES[player_i]],
                 row[VELOCITYZ_COLUMN_NAMES[player_i]]))

    return all_player_position_as_vec3_list


def all_player_eyeAngles_to_direction_vec3(df: pd.DataFrame,
                                           index_label) -> List[vec3]:
    '''
        Calculates and returns all looking direction vectors of players in a labelled tick
        NOTE: Based on CS:GO/source coordinate system: https://developer.valvesoftware.com/wiki/Coordinates
        Vectors are calculated based on pitch and yaw of player
    '''

    #Pitch Tipping nose up and down
    #Yaw Tipping nose left and right
    # X is forward, Y is left/East and Z is up and down in CS:GO!

    all_player_eyeAngles_to_direction_vec3_list = []

    for player_i in range(10):
        pitchDeg = df.at[index_label, EYEANGLE_PITCH_COLUMN_NAMES[player_i]]
        yawDeg = df.at[index_label, EYEANGLE_YAW_COLUMN_NAMES[player_i]]

        pitch = math.radians(pitchDeg)
        yaw = math.radians(yawDeg)

        # COPIED FROM https://stackoverflow.com/a/10569719 BY Neil Forrester
        xzLen = math.cos(pitch)
        x = xzLen * math.cos(yaw)
        y = xzLen * math.sin(-yaw)
        z = math.sin(pitch)  #Up and down

        all_player_eyeAngles_to_direction_vec3_list.append(vec3(x, y, z))

    return all_player_eyeAngles_to_direction_vec3_list


def extract(cond, x):
    if isinstance(x, numbers.Number):
        return x
    else:
        return np.extract(cond, x)


FARAWAY = 1.0e39
VELOCITYX_COLUMN_NAMES = data_loader.get_feature_column_names('VelocityX')
VELOCITYY_COLUMN_NAMES = data_loader.get_feature_column_names('VelocityY')
VELOCITYZ_COLUMN_NAMES = data_loader.get_feature_column_names('VelocityZ')

EYEANGLE_PITCH_COLUMN_NAMES = data_loader.get_feature_column_names(
    'EyeAnglePitch')
EYEANGLE_YAW_COLUMN_NAMES = data_loader.get_feature_column_names('EyeAngleYaw')

AIM_ON_ENEMY_COLUMN_NAMES = data_loader.get_feature_column_names(
    data_loader.get_feature_names_from_feature_subset("one_hot_aim_on_enemy"))

if __name__ == "__main__":
    sample = data_loader.load_sample_csv_as_df()

    l = all_player_eyeAngles_to_direction_vec3(sample, sample.index[0])
    pass
