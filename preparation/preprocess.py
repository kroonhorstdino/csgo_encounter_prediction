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

FEATURES_INFO = data_loader.load_config(
    Path('preparation') / 'features_info.json')

EYEANGLES_COLUMN_NAMES = data_loader.get_feature_column_names(
    'EyeAnglePitch') + data_loader.get_feature_column_names('EyeAngleYaw')


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


#TODO: Add one hot angles
def add_one_hot_encoding_angles(df: pd.DataFrame):
    '''
        Adds one hot encoding for angle to player (is a raycast within radius of enemy player)
        Removes 'EyeAnglesPitch' and 'EyeAnglesYaw' columns for each player
    '''
    '''
    def extract(cond, x):
    if isinstance(x, numbers.Number):
        return x
    else:
        return np.extract(cond, x)

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
        return vec3(extract(cond, self.x),
                    extract(cond, self.y),
                    extract(cond, self.z))
    def place(self, cond):
        r = vec3(np.zeros(cond.shape), np.zeros(cond.shape), np.zeros(cond.shape))
        np.place(r.x, cond, self.x)
        np.place(r.y, cond, self.y)
        np.place(r.z, cond, self.z)
        return r
rgb = vec3

(w, h) = (400, 300)         # Screen size
L = vec3(5, 5., -10)        # Point light position
E = vec3(0., 0.35, -1.)     # Eye position
FARAWAY = 1.0e39            # an implausibly huge distance

def raytrace(O, D, scene, bounce = 0):
    # O is the ray origin, D is the normalized ray direction
    # scene is a list of Sphere objects (see below)
    # bounce is the number of the bounce, starting at zero for camera rays

    distances = [s.intersect(O, D) for s in scene]
    nearest = reduce(np.minimum, distances)
    color = rgb(0, 0, 0)
    for (s, d) in zip(scene, distances):
        hit = (nearest != FARAWAY) & (d == nearest)
        if np.any(hit):
            dc = extract(hit, d)
            Oc = O.extract(hit)
            Dc = D.extract(hit)
            cc = s.light(Oc, Dc, dc, scene, bounce)
            color += cc.place(hit)
    return color

class Sphere:
    def __init__(self, center, r, diffuse, mirror = 0.5):
        self.c = center
        self.r = r
        self.diffuse = diffuse
        self.mirror = mirror

    def intersect(self, O, D):
        b = 2 * D.dot(O - self.c)
        c = abs(self.c) + abs(O) - 2 * self.c.dot(O) - (self.r * self.r)
        disc = (b ** 2) - (4 * c)
        sq = np.sqrt(np.maximum(0, disc))
        h0 = (-b - sq) / 2
        h1 = (-b + sq) / 2
        h = np.where((h0 > 0) & (h0 < h1), h0, h1)
        pred = (disc > 0) & (h > 0)
        return np.where(pred, h, FARAWAY)

    def diffusecolor(self, M):
        return self.diffuse

    def light(self, O, D, d, scene, bounce):
        M = (O + D * d)                         # intersection point
        N = (M - self.c) * (1. / self.r)        # normal
        toL = (L - M).norm()                    # direction to light
        toO = (E - M).norm()                    # direction to ray origin
        nudged = M + N * .0001                  # M nudged to avoid itself

        # Shadow: find if the point is shadowed or not.
        # This amounts to finding out if M can see the light
        light_distances = [s.intersect(nudged, toL) for s in scene]
        light_nearest = reduce(np.minimum, light_distances)
        seelight = light_distances[scene.index(self)] == light_nearest

        # Ambient
        color = rgb(0.05, 0.05, 0.05)

        # Lambert shading (diffuse)
        lv = np.maximum(N.dot(toL), 0)
        color += self.diffusecolor(M) * lv * seelight

        # Reflection
        if bounce < 2:
            rayD = (D - N * 2 * D.dot(N)).norm()
            color += raytrace(nudged, rayD, scene, bounce + 1) * self.mirror

        # Blinn-Phong shading (specular)
        phong = N.dot((toL + toO).norm())
        color += rgb(1, 1, 1) * np.power(np.clip(phong, 0, 1), 50) * seelight
        return color

class CheckeredSphere(Sphere):
    def diffusecolor(self, M):
        checker = ((M.x * 2).astype(int) % 2) == ((M.z * 2).astype(int) % 2)
        return self.diffuse * checker

scene = [
    Sphere(vec3(.75, .1, 1.), .6, rgb(0, 0, 1)),
    Sphere(vec3(-.75, .1, 2.25), .6, rgb(.5, .223, .5)),
    Sphere(vec3(-2.75, .1, 3.5), .6, rgb(1., .572, .184)),
    CheckeredSphere(vec3(0,-99999.5, 0), 99999, rgb(.75, .75, .75), 0.25),
    ]
    '''

    df.drop(columns=EYEANGLES_COLUMN_NAMES, inplace=True)
    return df


def add_one_hot_encoding_weapons(df: pd.DataFrame):
    '''
        Adds one hot encoding for weapons of player. All knifes are mapped to id 42
        Removes 'CurrentWeapon' columns for each player
    '''

    actual_column_names = data_loader.get_one_hot_encoded_weapon_feature_names(
    )

    weapon_id_df = pd.DataFrame(columns=actual_column_names,
                                index=df.index,
                                dtype=np.float32)

    weapon_id_df.sort_index(axis=1, inplace=True)

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

            #weapon_id_df[weapon_id_column_name].loc[index] = 1  #Weapon is used at this point in time for this player
            weapon_id_df.at[index, weapon_id_column_name] = 1

    weapon_id_df.fillna(0, inplace=True)

    current_weapon_columns_names = data_loader.get_feature_column_names(
        'CurrentWeapon')
    df.drop(columns=current_weapon_columns_names, inplace=True)

    df = pd.concat([df, weapon_id_df], sort=True)

    return df


if __name__ == "__main__":
    '''add_one_hot_encoding_weapons(
        data_loader.load_csv_as_df(
            Path('parsed_files/sprout-vs-ex-epsilon-m3-overpass.csv')))'''
    pass
