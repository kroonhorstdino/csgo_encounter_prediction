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

PATH_RESULTS = Path('results')
CONFIG_PATH = Path('config')

#TRAIN_CONFIG_PATH = Path('config' / 'train_config.json')
DATASET_CONFIG_PATH = Path('config') / 'dataset_config.json'

#DATASET_CONFIG = data_loader.load_json(DATASET_CONFIG_PATH)

FEATURES_INFO_PATH = Path('config') / 'features_info.json'


class CounterStrikeTestset(Dataset):
    def __init__(self, train_config, match_path):

        self.map = train_config["training"]["map"]

        self.testset_files = data_loader.load_h5_as_df(match_path, False)

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
        '''
        match_file = self.testset_files[index]
        match_df = data_loader.load_h5_as_df(
            match_file, False, column_names=self.all_column_names)
        '''
        match_df = self.testset_files

        player_features, classification_labels = data_loader.split_data_into_minibatch(
            match_df)

        return player_features, classification_labels, match_df.index.to_numpy(
        )

    def __len__(self):
        return len([self.testset_files])


def test_on_match(run_name: str, match_path, train_config_path):

    print("Start testing process...")

    #log_every_num_batches = 250 - (50 * min((4, args.verbose)))

    train_config = data_loader.load_json(train_config_path)
    DATASET_CONFIG = data_loader.load_json(DATASET_CONFIG_PATH)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    print("Using device: ", device)

    #OptimizerType = torch.optim.Adam

    #dataset_files_partition = prepare_dataset.get_dataset_partitions(DATASET_CONFIG["paths"]["testing_files_path"], [0.8, 0.2, 0])

    # the dataset returns a batch when called (because we get the whole batch from one file), the batch size of the data loader thus is set to 1 (default)
    # epoch size is how many elements the iterator of the generator will provide, NOTE should not be too small, because it have a significant overhead p=0.05
    test_set = CounterStrikeTestset(train_config, match_path)
    test_generator = torch.utils.data.DataLoader(test_set,
                                                 batch_size=1,
                                                 shuffle=False)

    print("(Mini-)batches per epoch: " + str(test_set.__len__()))

    model = data_loader.load_model_to_test(
        epoch_i=100,
        model_full_name=run_name,
        num_all_player_features=test_set.num_all_player_features)

    model.to(device)
    print("Model loaded")

    print("Creating tensorboard summary writer")

    writer = SummaryWriter(str(Path(PATH_RESULTS / run_name)))
    #writer.add_hparams({"test_set": "dummy_value"}, {})

    #criterion = nn.CrossEntropyLoss()
    binary_classification_loss = torch.nn.BCELoss()

    all_test_losses = []
    all_test_accuracies = []
    all_test_target_accuracies = []
    all_test_die_notdie_accuracies = []
    all_test_per_sec_accuracies = [[] for _ in range(20)]
    all_test_per_sec_predictions = [[] for _ in range(20)]
    all_test_per_sec_predictions_std = [[] for _ in range(20)]

    all_validation_losses = []
    all_validation_accuracies = []

    all_validation_roc_scores = []
    all_validation_pr_scores = []
    '''
        #####
        NOTE: TRAINING LOOP
        #####
    '''

    #
    ''' PROGRESS BARS '''
    epoch_display_stats = {
        "loss": 100.0,
        "accuracy": 0.0,
        "die_accuracy": 0.0,
        "not_die_accuracy": 0.0
    }

    test_prog_bar = tqdm(total=test_set.__len__(),
                         desc="MATCH:",
                         dynamic_ncols=True)
    epoch_prog_bar = tqdm(desc="Match testing progress",
                          postfix=epoch_display_stats,
                          dynamic_ncols=True)

    now = time.time()
    test_prog_bar.set_description(f"MATCH 0")

    # reset seed   https://github.com/pytorch/pytorch/issues/5059  data loader returns the same values
    np.random.seed()

    #epoch_per_sec_accuracies = [[] for _ in range(20)]
    #epoch_per_sec_predictions = [[] for _ in range(20)]

    #My acc tracking
    epoch_loss_all_player = []

    epoch_accuracy_all_player = []

    epoch_accuracy_die = []
    epoch_accuracy_not_die = []

    #Iterate through all matches
    for match_i, (all_X, all_y, match_indecies) in enumerate(test_generator):

        #Input and labels from match
        all_X = [(player_X[0, :]).to(device) for player_X in all_X]
        all_y = (all_y[0, :]).to(device)
        match_indecies = (match_indecies[0, :]).to(device)

        output = model(all_X)
        #TODO: Maybe nn.BCEWithLogitsLoss?
        output = torch.sigmoid(output)
        all_y_np = all_y.cpu().detach().numpy()
        output_np = output.cpu().detach().numpy()

        batch_loss_all_player = binary_classification_loss(
            output, all_y).cpu().detach().numpy().astype(np.float32)
        epoch_loss_all_player.append(batch_loss_all_player)

        #
        '''
            NOTE: Log accuracy values 
        '''

        # Acc values for all players and for targeted player
        batch_accuracy_all_player = ((output > 0.5) == (
            all_y > 0.5)).cpu().numpy().astype(np.float32)

        batch_accuracy_all_player_mean = batch_accuracy_all_player.mean()

        # Accuracy for predicting deaths correctly and survival
        batch_accuracy_die = ((output > 0.5) == (all_y > 0.5)).view(-1)[
            all_y.view(-1) > 0.5].cpu().numpy().reshape(-1).astype(np.float32)
        batch_accuracy_not_die = ((output > 0.5) == (all_y > 0.5)).view(-1)[
            all_y.view(-1) < 0.5].cpu().numpy().reshape(-1).astype(np.float32)

        epoch_accuracy_all_player.append(
            batch_accuracy_all_player_mean
        )  # Add mean accuracy to epoch accuracies

        # these have varying size, so calculating the proper mean across batches takes more work

        epoch_prog_bar.update()

        #Save result for later analyzation

        match_path = Path(match_path)

        match_indices_np = match_indecies.cpu().detach().numpy()

        np.save(
            f'results/{run_name}/{match_path.with_suffix("").name}_match_indecies.npy',
            match_indices_np)
        np.save(
            f'results/{run_name}/{match_path.with_suffix("").name}_predictions_and_labels.npy',
            np.array([output_np, all_y_np]))
        np.save(
            f'results/{run_name}/{match_path.with_suffix("").name}_die_not_die_acc.npy',
            np.array([batch_accuracy_die, batch_accuracy_not_die]))
        '''np.save(
            f'results/{run_name}/{match_path.with_suffix("")}_accuracy.npy',
            np.array([
                epoch_accuracy_all_player,
                match_indecies.cpu().detach().numpy()
            ]))'''

        writer.add_pr_curve(
            f'Testing/{match_path.with_suffix("").name}/PR_Curve',
            output_np,
            all_y_np,
            global_step=None,
            num_thresholds=127,
            weights=None,
            walltime=None)

    epoch_prog_bar.reset()
    '''
    writer.add_scalars(
        "Training/Loss", {
            'Loss all players': np.array(epoch_loss_all_player).mean(),
            'Loss target player':
            np.array(epoch_loss_target_player).mean()
        }, epoch_i)

    writer.add_scalars(
        "Training/Accuracy", {
            "Accuracy all players":
            np.array(epoch_accuracy_all_player).mean(),
            "Accuracy target player":
            np.array(epoch_accuracy_target_player).mean()
        }, epoch_i)

    writer.add_scalars(
        "Training/Die and not die accuracy", {
            "Accuracy for Death":
            np.array(epoch_accuracy_die).mean(),
            "Accuracy for Survival (not die)":
            np.array(epoch_accuracy_not_die).mean()
        }, epoch_i)
    '''
    '''
    all_test_losses.append(np.array(epoch_loss_all_player).mean())
    all_test_accuracies.append(np.array(epoch_accuracy_all_player).mean())
    all_test_target_accuracies.append(
        np.array(epoch_accuracy_target_player).mean())
    all_test_die_notdie_accuracies.append(
        (np.array(epoch_accuracy_die).mean(),
            np.array(epoch_accuracy_not_die).mean()))
    '''

    sys.stdout.flush()

    #One game parsed
    test_prog_bar.update()

    writer.close()


if __name__ == '__main__':
    test_on_match(
        'first_exp_0',
        '/home/hueter/csgo_dataset/processed_files/24bhb-vs-shift-m2-nuke.h5',
        'config/first_exp/train_config_first_exp_0.json')
