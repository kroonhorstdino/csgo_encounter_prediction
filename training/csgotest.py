import matplotlib.animation as animation
from torch.utils.data.dataset import Dataset
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn as nn
import torch

from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from pydoc import locate

from tqdm import tqdm

import commentjson
from termcolor import colored
import os
import random
import glob
import itertools
import sys
import time
from pathlib import Path
from typing import List, Tuple

sys.path.insert(0, str(Path.cwd() / 'preparation/'))

import models
import preprocess
import data_loader


class CounterStrikeDataset(Dataset):
    def __init__(self, data_config, train_config, isValidationSet=False):
        if (not isValidationSet):
            self.dataset_files = data_loader.get_files_in_directory(
                data_config["paths"]["training_files_path"], '.h5')
        else:
            self.dataset_files = data_loader.get_files_in_directory(
                data_config["paths"]["validation_files_path"], '.h5')

        # Small sample of dataset
        dataset_sample = data_loader.load_h5_as_df(self.dataset_files[0],
                                                   False)
        '''
            Chunks -> randomized files. Always size of a power of 2
            Batches -> Also always power of 2
        '''

        self.batch_row_size = train_config["training"]["batch_size"]

        self.num_chunks = len(self.dataset_files)
        # Num of rows for each chunk
        self.chunks_row_size = data_config["randomization"]["chunk_row_size"]
        # Num of batches in the entire dataset
        self.num_batches = int(
            (self.num_chunks * self.chunks_row_size) / self.batch_row_size)
        # Num of batches per chunk in dataset
        self.batches_per_chunk = int(self.num_batches / self.num_chunks)

        self.num_epoch = train_config["training"]["num_epoch"]

        self.death_time_window = data_config["preprocessing"][
            "death_time_window"]

        self.num_players = 10
        self.num_all_player_features = data_loader.get_num_player_features(
            dataset_sample.columns)

    def __getitem__(self, index):

        player_i = random.randrange(0, 10)

        #DEBUG:
        #index = 0

        chunk_file = self.dataset_files[self.batch_index_to_chunk_index(index)]
        #start_index, end_index = self.get_indicies_in_chunk(index) #NOTE: If you want a specific area of chunk. Only without balancing during loading!

        chunk = data_loader.load_h5_as_df(chunk_file, True)
        # .iloc[start_index:end_index] A batch is going to be loaded from this chunk >batches_per_chunk< times anyways. If random, may be not so bad NOTE: FIXME:

        player_features, classification_labels = data_loader.get_minibatch_balanced_player(
            chunk, player_i, batch_size=self.batch_row_size)

        return player_features, classification_labels, player_i

    def batch_index_to_chunk_index(self, batch_index: int):
        ''' In what file/chunk is this batch '''

        return int(batch_index / self.batches_per_chunk)

    def get_indicies_in_chunk(self, batch_index) -> Tuple[int, int]:
        batch_in_chunk = batch_index % self.batches_per_chunk

        start_index_in_chunk = batch_in_chunk * self.batch_row_size
        # End index in selecting is exclusive in pandas
        end_index_in_chunk = start_index_in_chunk + self.batch_row_size

        return start_index_in_chunk, end_index_in_chunk

    def __len__(self):
        return self.batches_per_chunk * self.num_chunks


def is_validation_epoch(epoch_i: int):
    return epoch_i % 3 == 0


def train_csgo(dataset_config_path: Path, train_config_path: Path):

    print("Start training...")

    #TODO:
    log_every_num_batches = 50

    train_config = data_loader.load_config(train_config_path)
    dataset_config = data_loader.load_config(dataset_config_path)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    print("Using device: ", device)

    OptimizerType = torch.optim.Adam

    # the dataset returns a batch when called (because we get the whole batch from one file), the batch size of the data loader thus is set to 1 (default)
    # epoch size is how many elements the iterator of the generator will provide, NOTE should not be too small, because it have a significant overhead p=0.05
    training_set = CounterStrikeDataset(dataset_config,
                                        train_config,
                                        isValidationSet=False)
    training_generator = torch.utils.data.DataLoader(training_set,
                                                     batch_size=1,
                                                     shuffle=False)

    validation_set = CounterStrikeDataset(dataset_config,
                                          train_config,
                                          isValidationSet=True)
    validation_generator = torch.utils.data.DataLoader(validation_set,
                                                       batch_size=1,
                                                       shuffle=True)

    print("(Mini-)batches per epoch: " + str(training_set.__len__()))

    model = models.SharedWeightsCSGO(
        num_all_player_features=training_set.num_all_player_features,
        num_labels=10)

    model.to(device)

    #criterion = nn.CrossEntropyLoss()
    binary_classification_loss = torch.nn.BCELoss()
    optimizer = OptimizerType(model.parameters(),
                              lr=train_config["training"]["lr"])

    all_train_losses = []
    all_train_accuracies = []
    all_train_target_accuracies = []
    all_train_die_notdie_accuracies = []
    all_train_per_sec_accuracies = [[] for _ in range(20)]
    all_train_per_sec_predictions = [[] for _ in range(20)]
    all_train_per_sec_predictions_std = [[] for _ in range(20)]

    all_validation_losses = []
    all_validation_accuracies = []

    all_validation_roc_scores = []
    all_validation_pr_scores = []
    '''
        #####
        NOTE: TRAINING LOOP
        #####
    '''

    epoch_display_stats = {
        "loss": 100.0,
        "accuracy": 0.0,
        "die_accuracy": 0.0,
        "not_die_accuracy": 0.0
    }

    train_prog_bar = tqdm(total=training_set.num_epoch, desc="EPOCH 0")
    validation_prog_bar = tqdm(desc="Validation epoch",
                               total=len(validation_generator))
    epoch_prog_bar = tqdm(total=training_set.num_batches,
                          desc="Epoch progress",
                          postfix=epoch_display_stats)

    for epoch_i in range(training_set.num_epoch):

        now = time.time()

        # reset seed   https://github.com/pytorch/pytorch/issues/5059  data loader returns the same values
        np.random.seed()

        epoch_per_sec_accuracies = [[] for _ in range(20)]
        epoch_per_sec_predictions = [[] for _ in range(20)]

        #My acc tracking
        epoch_loss_all_player = []
        epoch_loss_target_player = []

        epoch_accuracy_all_player = []
        epoch_accuracy_target_player = []

        epoch_accuracy_die = []
        epoch_accuracy_not_die = []

        for batch_i, (X, y, player_i) in enumerate(training_generator):
            # since we get a batch of size 1 of batch of real batch size, we take the 0th element

            #death_times = death_times[0]

            # training_generator adds one dimension to each tensor, so we have to extract the data
            X = [(hero_X[0, :]).to(device) for hero_X in X]
            y = (y[0, :]).to(device)
            player_i = player_i[0].to(device)

            # Forward + Backward + Optimize
            # Remove gradients from last iteration
            optimizer.zero_grad()

            #DEBUG:print(X[0][0])
            '''
                NOTE: Forward pass
            '''

            output = model(X)
            output = torch.sigmoid(output)
            output_np = output.cpu().detach().numpy()

            # only backpropagate the loss for player_i (so the training data is balanced)
            player_i_output = output[:, player_i]
            player_i_labels = y[:, player_i]
            '''
                NOTE: Loss calculation / Backpropagation / Weight adjustment
            '''

            # Calculate loss only for the output of one player
            batch_loss_target_player = binary_classification_loss(
                player_i_output, player_i_labels)

            batch_loss_target_player.backward()
            optimizer.step()

            # Log loss
            epoch_loss_target_player.append(batch_loss_target_player)

            batch_loss_all_player = binary_classification_loss(
                output, y).cpu().detach().numpy().astype(np.float32)
            epoch_loss_all_player.append(batch_loss_all_player)

            #
            '''
                NOTE: Log accuracy values 
            '''

            #TODO: Obs
            # Acc values for all players and for targeted player
            batch_accuracy_all_player = ((output > 0.5) == (
                y > 0.5)).cpu().numpy().astype(np.float32)
            batch_accuracy_target_player = ((player_i_output > 0.5) == (
                y[:, player_i] > 0.5)).cpu().numpy().reshape(-1).astype(
                    np.float32)

            #FIXME: Look into this black magic!
            batch_accuracy_die = ((output > 0.5) == (y > 0.5)).view(-1)[
                y.view(-1) > 0.5].cpu().numpy().reshape(-1).astype(np.float32)
            batch_accuracy_not_die = ((output > 0.5) == (y > 0.5)).view(-1)[
                y.view(-1) < 0.5].cpu().numpy().reshape(-1).astype(np.float32)

            epoch_accuracy_all_player.append(batch_accuracy_all_player.mean(
            ))  # Add mean accuracy to epoch accuracies
            epoch_accuracy_target_player.append(
                batch_accuracy_target_player.mean())

            # these have varying size, so calculating the proper mean across batches takes more work
            epoch_accuracy_die.extend(batch_accuracy_die)
            epoch_accuracy_not_die.extend(batch_accuracy_not_die)

            #
            '''
                NOTE: Print accuracies and losses
            '''
            if (batch_i >= log_every_num_batches) and (
                    batch_i % log_every_num_batches) == 0:

                # Only get accuracy of the batches after the last logging of accuracies
                last_batches_loss = np.array(
                    epoch_loss_all_player[-log_every_num_batches:]).mean()
                last_batches_accuracy = np.array(
                    epoch_accuracy_all_player[-log_every_num_batches:]).mean()

                last_batches_die_acc = np.array(
                    epoch_accuracy_die[-49:]).mean()
                last_batches_not_die_acc = np.array(
                    epoch_accuracy_not_die[-49:]).mean()

                epoch_prog_bar.write(
                    f'E {epoch_i}, B {batch_i} -- Loss: {last_batches_loss} -- Acc: {last_batches_accuracy}'
                )
                epoch_prog_bar.write(
                    f'die acc: {last_batches_die_acc} -- not die acc: {last_batches_not_die_acc}'
                )

                epoch_prog_bar.set_postfix({
                    "loss":
                    last_batches_loss,
                    "accuracy":
                    last_batches_accuracy,
                    "die_accuracy":
                    last_batches_die_acc,
                    "not_die_accuracy":
                    last_batches_not_die_acc
                })

                sys.stdout.flush()

            epoch_prog_bar.update()
        '''
            #####
            NOTE: EPOCH FINISHED
            #####
        '''

        epoch_prog_bar.reset()

        if (epoch_i % 10) == 9:
            np.save('epoch_per_sec_predictions.npy',
                    np.array(epoch_per_sec_predictions))

        all_train_losses.append(np.array(epoch_loss_all_player).mean())
        all_train_accuracies.append(np.array(epoch_accuracy_all_player).mean())
        all_train_target_accuracies.append(
            np.array(epoch_accuracy_target_player).mean())
        all_train_die_notdie_accuracies.append(
            (np.array(epoch_accuracy_die).mean(),
             np.array(epoch_accuracy_not_die).mean()))
        '''
        TODO: For continoos death labels

        for timeslot_i in range(20):
            all_train_per_sec_accuracies[timeslot_i].append(
                np.array(epoch_per_sec_accuracies[timeslot_i]).mean())
            all_train_per_sec_predictions[timeslot_i].append(
                np.array(epoch_per_sec_predictions[timeslot_i]).mean())
            all_train_per_sec_predictions_std[timeslot_i].append(
                np.array(epoch_per_sec_predictions[timeslot_i]).std())
        '''
        '''
            #####
            NOTE: VALIDATION EPOCH
            #####
        '''

        if (is_validation_epoch(epoch_i)):

            epoch_loss_all_player = []
            epoch_accuracy_all_player = []

            epoch_all_pred = []
            epoch_all_y = []

            with torch.no_grad():
                for X, y, player_i in validation_generator:
                    X = [(hero_X[0, :]).to(device) for hero_X in X]
                    y = (y[0, :]).to(device)

                    output = model(X)
                    output = torch.sigmoid(output)
                    output_np = output.cpu().detach().numpy()

                    epoch_loss_all_player.append(
                        binary_classification_loss(
                            output,
                            y).cpu().detach().numpy().reshape(-1).mean())
                    accuracy_vec = ((output > 0.5) == (
                        y > 0.5)).cpu().numpy().reshape(-1).astype(np.float32)
                    epoch_accuracy_all_player.append(accuracy_vec.mean())

                    epoch_all_pred.extend(output_np.reshape(-1))
                    epoch_all_y.extend(y.cpu().numpy().reshape(-1))

                    validation_prog_bar.update()

            all_validation_roc_scores.append(
                roc_auc_score(epoch_all_y, epoch_all_pred))
            all_validation_pr_scores.append(
                average_precision_score(epoch_all_y, epoch_all_pred))

            all_validation_losses.append(
                np.array(epoch_loss_all_player).mean())
            all_validation_accuracies.append(
                np.array(epoch_accuracy_all_player).mean())

        else:
            # just copy the previous validation statistics, so we can plot it togeather with training statistics
            all_validation_losses.append(all_validation_losses[-1])
            all_validation_accuracies.append(all_validation_accuracies[-1])
            all_validation_roc_scores.append(all_validation_roc_scores[-1])
            all_validation_pr_scores.append(all_validation_pr_scores[-1])

        validation_prog_bar.reset()

        #
        '''
            #####
            NOTE: VALIDATION EPOCH END
            #####
        '''

        train_prog_bar.write(
            f"Epoch done {epoch_i} | loss: {np.array(epoch_loss_all_player).mean()} | accuracy: {np.array(epoch_accuracy_target_player).mean()}"
        )
        train_prog_bar.write(f"Epoch took: {time.time() - now}")
        sys.stdout.flush()

        if (epoch_i % 10) == 9:
            np.save('all_train_per_sec_predictions.npy',
                    np.array(all_train_per_sec_predictions))
            np.save('all_train_per_sec_predictions_std.npy',
                    np.array(all_train_per_sec_predictions_std))

        if (epoch_i % 100) == 99:
            torch.save(model.state_dict(), "model" + str(epoch_i) + ".model")

        train_prog_bar.update()
        train_prog_bar.set_description(f"EPOCH {epoch_i}")


if __name__ == "__main__":
    train_csgo('config/dataset_config.json', 'config/train_config.json')
