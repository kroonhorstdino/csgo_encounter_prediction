import numpy as np
import pandas as pd
import time
import numbers
import math
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import (average_precision_score, precision_recall_curve,
                             roc_auc_score, roc_curve)

import sys
import os
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path.cwd() / 'preparation/'))
sys.path.insert(0, str(Path.cwd() / 'training/'))

import data_loader
import models
import preprocess
import csgotest
import models
import prepare_dataset

PATH_RESULTS = Path('test_results')
CONFIG_PATH = Path('config')

#TRAIN_CONFIG_PATH = Path('config' / 'train_config.json')
DATASET_CONFIG_PATH = Path('config') / 'dataset_config.json'

#DATASET_CONFIG = data_loader.load_json(DATASET_CONFIG_PATH)

FEATURES_INFO_PATH = Path('config') / 'features_info.json'


class CounterStrikeTestset(Dataset):
    def __init__(self, train_config, match_path):

        self.map = train_config["training"]["map"]

        self.testset_files = data_loader.load_feather_as_df(match_path, False)

        self.feature_set = train_config["training"]["feature_set"]
        self.label_set = train_config["training"]["label_set"]
        self.features_column_names = data_loader.get_column_names_from_features_set(
            self.feature_set)
        self.labels_column_names = data_loader.get_die_within_seconds_column_names(
        )
        self.all_column_names = self.features_column_names + self.labels_column_names

        # Small sample of dataset
        testset_sample = self.testset_files.iloc[0:10]
        self.num_players = 10
        self.num_all_player_features = data_loader.get_num_player_features(
            testset_sample.columns)

    def __getitem__(self, index):
        match_df = self.testset_files

        player_features, classification_labels = data_loader.split_data_into_minibatch(
            match_df)

        return player_features, classification_labels, match_df.index.to_numpy(
        )

    def __len__(self):
        return len([self.testset_files])


def test_on_match(run_name: str, epoch: int, match_path, train_config_path):
    '''
        run_name --> Name of the run the model was trained with
    '''

    print("Start testing process...")

    if (not Path.exists(PATH_RESULTS / run_name)):
        os.mkdir(str(PATH_RESULTS / run_name))

    #log_every_num_batches = 250 - (50 * min((4, args.verbose)))

    train_config = data_loader.load_json(train_config_path)
    DATASET_CONFIG = data_loader.load_json(DATASET_CONFIG_PATH)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    print("Using device: ", device)

    feature_set = train_config["training"]["feature_set"]
    label_set = train_config["training"]["label_set"]
    features_column_names = data_loader.get_column_names_from_features_set(
        feature_set)
    labels_column_names = data_loader.get_die_within_seconds_column_names()
    all_column_names = features_column_names + labels_column_names

    testfiles = data_loader.get_files_in_directory(
        Path(DATASET_CONFIG["paths"]["test_files_path"]), 'inferno.feather')

    writer = SummaryWriter(str(Path(PATH_RESULTS / run_name)))
    #writer.add_hparams({"test_set": "dummy_value"}, {})
    num_all_player_features = len(
        data_loader.get_column_names_from_features_set(
            train_config["training"]["feature_set"]))

    model = data_loader.load_model_to_test(
        epoch_i=epoch,
        model_full_name=run_name,
        num_all_player_features=num_all_player_features)

    model.to(device)
    print("Model loaded")

    #criterion = nn.CrossEntropyLoss()
    #binary_classification_loss = torch.nn.BCELoss()
    print("Creating tensorboard summary writer")

    all_output_np = []
    all_pred_np = []
    all_match_indices = []

    for match_id in range(len(testfiles)):

        df = data_loader.load_feather_as_df(testfiles[match_id],
                                            False,
                                            column_names=all_column_names +
                                            ['Tick', 'Time', 'Round'])

        #LOAD THE MATCH DATA

        all_X, all_y = data_loader.split_data_into_minibatch(df)
        match_indecies = np.array(df.index.values)

        #Input and labels from match
        all_X = torch.from_numpy(np.array(all_X)).to(device)
        all_y = torch.from_numpy(np.array(all_y)).to(device)

        output = model(all_X)
        #TODO: Maybe nn.BCEWithLogitsLoss?
        output = torch.sigmoid(output)
        all_y_np = all_y.cpu().detach().numpy()
        output_np = output.cpu().detach().numpy()

        print(average_precision_score(all_y_np, output_np))

        match_indices_np = match_indecies

        all_pred_np.append(all_y_np)
        all_output_np.append(output_np)
        all_match_indices.append(match_indices_np)

        writer.add_pr_curve(f'Testing/{run_name}/PR_Curve',
                            output_np.flatten(), all_y_np.flatten(), match_id)
        sys.stdout.flush()

        print(
            f"Testing on match {Path(testfiles[match_id]).name} with model {run_name}_{epoch} and config {train_config_path} finished"
        )

    np.save(f'{str(PATH_RESULTS / run_name)}_predictions_and_labels.npy',
            np.array([np.array(all_output_np),
                      np.array(all_pred_np)]))
    np.save(f'{str(PATH_RESULTS / run_name)}_match_indecies.npy',
            np.array(all_match_indices))

    writer.close()
    print("Finished on all matches")


if __name__ == '__main__':
    test_on_match(
        'big_data_8', 41,
        '/home/hueter/csgo_dataset/test_data/astralis-vs-cloud9-inferno.feather',
        'config/third_exp/train_config_third_exp_4.json')
