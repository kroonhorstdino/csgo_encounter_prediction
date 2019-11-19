import numpy as np
import pandas as pd

import sys
import json

from pathlib import Path
from typing import Tuple, List, Union

import glob
import random

import torch
'''
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset
'''
sys.path.insert(0, str(Path.cwd() / 'training/'))

import models


def get_minibatch_balanced_player(
        data: pd.DataFrame,
        player_i,
        batch_size=128,
        max_time_to_next_death=5,
) -> Tuple[List[pd.DataFrame], pd.DataFrame]:

    player_dies_mask, player_not_die_mask = get_player_minibatch_mask(
        data, player_i, 5)

    num_sample_from_die = int(batch_size * 0.5)
    num_sample_from_not_die = batch_size - num_sample_from_die

    # Player stays dead
    have_enough_unique_data = sum(player_dies_mask) > num_sample_from_die

    data_batch_die = data.loc[player_dies_mask].sample(
        n=num_sample_from_die, replace=(have_enough_unique_data == False))

    # Player stays alive
    have_enough_unique_data = sum(
        player_not_die_mask) > num_sample_from_not_die
    data_batch_not_die = data.loc[player_not_die_mask].sample(
        n=num_sample_from_not_die, replace=(have_enough_unique_data == False))

    data_batch = pd.concat([data_batch_die, data_batch_not_die])

    return split_data_into_minibatch(data_batch, max_time_to_next_death)


def get_player_minibatch_mask(df: pd.DataFrame,
                              player_i: int,
                              max_time_to_next_death: int = 5,
                              is_binary=True):
    '''
        isAlive state labelling:
            0: is dead
            1: is alive

        Death state labelling:
            0: is not dead, and will stay alive
            1: has died within time window, "isDying"
            2: is dead, longer than time window

        WARN: This may remove the death label 2 from the specific player, but all other players may still have labelling 2.
    '''

    classification_clmn_name = get_die_within_seconds_column_names(
        10)[player_i]
    isAlive_clmn_name = get_isAlive_column_names(10)[player_i]

    isAlive_mask = df[isAlive_clmn_name] == 1
    isDying_mask = df[classification_clmn_name] == 1
    isLongDead_mask = df[classification_clmn_name] == 2

    # Instances where player is going to die, meaning being alive and dead in the future
    player_die_mask = isAlive_mask & isDying_mask & ~isLongDead_mask
    # Player is alive and is either not going to die or not long dead already (last case should not be possible with correct data)
    player_not_die_mask = isAlive_mask & ~isDying_mask & ~isLongDead_mask

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

    #TODO: Best solution?
    ''' Remove all 2 labels (is dead longer than time window) and change it to zero (simply dead) for binary classification '''
    classification_labels[classification_labels > 1] = 1

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


def get_column_indices_from_names(df_columns,
                                  column_names: List[str]) -> List[int]:
    ''' TODO:
        Get the indecies of columns from their names
    '''
    pass


def get_column_names_from_features_set(feature_set: str):
    '''
        Generates a list of feature names for each player from a feature set.
        Labels appear as they would in the dataframe or during parsing
    '''

    actual_column_names = []
    all_feature_names = []

    for feature_subset_name in FEATURES_INFO["player_features_sets"][
            feature_set]:
        all_feature_names.extend(
            get_feature_names_from_feature_subset(feature_subset_name))

    for feature_name in all_feature_names:
        actual_column_names.extend(get_feature_column_names(feature_name))

    return actual_column_names


def get_one_hot_encoded_weapon_feature_names():

    weapon_one_hot_ids_column_name = []
    actual_column_names = []

    for index in ITEM_DEFINITION_INDEX_MAP['itemDefinitionIndexMap']:
        if (int(index) >= 500 or int(index) == 59):  # Indecies of other knifes
            continue

        weapon_one_hot_ids_column_name.append(index)

    weapon_one_hot_ids_column_name.sort(key=int)

    #For each feature name, generate feature name for each player
    for weapon_id in weapon_one_hot_ids_column_name:
        actual_column_names.extend(
            get_feature_column_names(f'Weapon_{weapon_id}'))

    return actual_column_names


def get_feature_names_from_feature_subset(feature_subset: str):
    '''
        Extracts feature names from subset in config. Does not generate names for each player.
    '''

    feature_name_list = []

    for feature_obj in FEATURES_INFO["player_features"][feature_subset]:
        feature_name_list.append(feature_obj["title"])

    return feature_name_list


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


def get_feature_column_names(feature_name: Union[str, list], num_players=10):
    actual_column_names = []

    # If a list of features go through each and store it as a list in a list
    if type(feature_name) == list:
        for feature_n in feature_name:
            one_feature_column_names = []

            for player_i in range(num_players):
                one_feature_column_names.append(f'f_{player_i}_{feature_n}')

            actual_column_names.append(one_feature_column_names)

    else:
        for player_i in range(num_players):
            actual_column_names.append(f'f_{player_i}_{feature_name}')

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


def load_csv_as_df(filePath: Path, cast_to_float: bool = True) -> pd.DataFrame:
    '''
        Loads parsed .csv file of match as df with all nessecary index modifications and NaN handling
        Removes Round column and sets Tick column as index for dataframe
    '''

    df = pd.DataFrame
    if (cast_to_float):
        df = pd.read_csv(filePath, sep=',').astype(np.float32)
        df.fillna(value=WEAPON_COLUMN_FILLNA_VALUES, inplace=True)
        df.fillna(0, inplace=True)
    else:
        df = pd.read_csv(filePath, sep=',', na_values='missing')

    df.set_index('Tick', inplace=True)
    # NOTE: Don't drop if still relevant
    #df.drop(columns=['Round'], inplace=True)

    return df


def load_feather_as_df(filePath: Path,
                       drop_ticks: bool,
                       key: str = 'player_info',
                       column_names: List[str] = None) -> pd.DataFrame:
    '''
        Loads a dataframe from .feather file with given key \n
        'drop_ticks' drops 'Tick' as index and removes that column from df reverting back to integer based index
    '''

    # DEBUG: print(filePath)
    df: pd.DataFrame
    if column_names is None:
        df = pd.read_feather(filePath).astype(np.float32)
    else:
        df = pd.read_feather(filePath, columns=column_names).astype(np.float32)

    if (not drop_ticks):
        #Set the ticks as index
        df.set_index(['Tick'], inplace=True)

    df.fillna(0.0, inplace=True)

    return df


def load_json(config_path: Path):
    config = None
    try:
        with open(str(config_path)) as json_file:
            print("Loading JSON file " + str(config_path))
            config = json.load(json_file)
    except:
        print("Couldn't load this JSON file!")
        raise

    return config


def load_sample_csv_as_df():
    '''
        Get a sample csv file as dataframe from parsed data
    '''
    return load_csv_as_df(
        random.choice(
            get_files_in_directory(
                DATASET_CONFIG["paths"]["parsed_files_path"], '.csv')))


def load_sample_h5_as_df(drop_ticks: bool, key: str = None):
    '''
        Get a sample h5 file as dataframe from training data
    '''
    return load_feather_as_df(random.choice(
        get_files_in_directory(DATASET_CONFIG["paths"]["training_files_path"],
                               '.csv')),
                              drop_ticks,
                              key=key)


def load_model_to_test(epoch_i: int = 100,
                       model_full_name: str = None,
                       num_all_player_features=None):
    MODELS_PATH = Path(f'models/{model_full_name}')

    checkpoint_path = str(MODELS_PATH /
                          f'model_{model_full_name}_EPOCH_{epoch_i}.model')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    checkpoint = torch.load(checkpoint_path, map_location=device)
    TRAIN_CONFIG = checkpoint["train_config"]

    #Recreate
    model = models.SharedWeightsCSGO(
        num_all_player_features,
        shared_layer_sizes=TRAIN_CONFIG["topography"]["shared_layers"],
        dense_layer_sizes=TRAIN_CONFIG["topography"]["dense_layers"])

    model.eval()
    model.to(device)

    model.load_state_dict(checkpoint['state_dict'])
    for parameter in model.parameters():
        parameter.requires_grad = False

    return model


def get_team_iterables():
    '''
        Get indecies for all players divided into two arrays representing the teams
    '''
    teams = np.split(np.arange(10), 2)

    return teams


def get_ally_team_iterable_index(player_i: int):
    '''
        Returns in which team the player_i is. Refers to team arrays generated by get_team_iterables
        Either 0 or 1 is returned
    '''
    return 0 if player_i < 5 else 1


def get_enemy_team_iterable_index(player_i: int):
    '''
        same as get_ally_team_iterable_index but for
    '''
    return (get_ally_team_iterable_index(player_i) - 1) * -1


DATASET_CONFIG = load_json('config/dataset_config.json')

FEATURES_INFO: dict = load_json('config/features_info.json')
ITEM_DEFINITION_INDEX_MAP: dict = load_json(
    'config/item_definition_index_map.json')

WEAPON_COLUMN_FILLNA_VALUES = dict(
    map(lambda name: (name, 42.0), get_feature_column_names('CurrentWeapon')))

MODEL_NAME_TEMPLATE = "model_{run_name}_EPOCH_{epoch_i}"

# Testing
if __name__ == "__main__":
    '''csv_file = get_files_in_directory('../../csgo_dataset/parsed_files',
                                      '.csv')[0]
    df = load_csv_as_df(csv_file)'''

    #a = [[get_ally_team_iterable_index(player_i),get_enemy_team_iterable_index(player_i)] for player_i in range(10)]
    #print(a)
    a = load_model_to_test(1000, 'exp_0', 2610)
    print(a)
