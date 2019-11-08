import argparse as ap
from datetime import date
import glob
import itertools
import os
import random
import sys
import time
from pathlib import Path
from pydoc import locate
from typing import List, Tuple

import commentjson
import matplotlib.animation as animation
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
from sklearn.metrics import (average_precision_score, precision_recall_curve,
                             roc_auc_score, roc_curve)
from termcolor import colored
from torch.autograd import Variable
from torch.utils.data.dataset import Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

sys.path.insert(0, str(Path.cwd() / 'preparation/'))

import data_loader
import models
import preprocess
import prepare_dataset

PATH_RESULTS = Path('results')

'''
    TODO: Download files in demo databse of lasse
    TODO: Jupyter test_angle_prep
    TODO: Get aim on enemy right
    TODO: Get feature selection right
    TODO: Train with all layers and big dataset
    TODO: Hyperparameter search
    TODO: Test on a match
'''

class CounterStrikeDataset(Dataset):
    def __init__(self,
                 data_config,
                 train_config,
                 dataset_files_partition,
                 isValidationSet=False):
        if (not isValidationSet):
            self.dataset_files = dataset_files_partition[0]
        else:
            self.dataset_files = dataset_files_partition[1]

        #TODO: Choose columns based on feature set in training config
        self.feature_set = train_config["training"]["feature_set"]
        self.label_set = train_config["training"]["label_set"]
        self.features_column_names = data_loader.get_column_names_from_features_set(
            self.feature_set)
        self.labels_column_names = data_loader.get_die_within_seconds_column_names(
        )
        self.all_column_names = self.features_column_names + self.labels_column_names

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

        # Small sample of dataset
        dataset_sample = data_loader.load_h5_as_df(self.dataset_files[0],
                                                   False,
                                                   self.all_column_names)
        '''
            Chunks -> randomized files. Always size of a power of 2
            Batches -> Also always power of 2
        '''

        self.num_players = 10
        self.num_all_player_features = data_loader.get_num_player_features(
            dataset_sample.columns)

    def __getitem__(self, index):

        player_i = random.randrange(0, 10)

        #DEBUG:
        #index = 0

        chunk_file = self.dataset_files[self.batch_index_to_chunk_index(index)]
        #start_index, end_index = self.get_indicies_in_chunk(index) #NOTE: If you want a specific area of chunk. Only without balancing during loading!

        chunk = data_loader.load_h5_as_df(chunk_file,
                                          True,
                                          column_names=self.all_column_names)
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


def train_csgo(dataset_config_path: Path,
               train_config_path: Path,
               run_name: str = None,
               loss_calculation_mode='target'):

    print("Start training process...")

    #TODO: Do something
    log_every_num_batches = 250 - (50 * min((4, args.verbose)))

    TRAIN_CONFIG = data_loader.load_json(train_config_path)
    DATASET_CONFIG = data_loader.load_json(dataset_config_path)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    print("Using device: ", device)

    OptimizerType = torch.optim.Adam

    dataset_files_partition = prepare_dataset.get_dataset_partitions(
        DATASET_CONFIG["paths"]["training_files_path"], [0.8, 0.2, 0])

    # the dataset returns a batch when called (because we get the whole batch from one file), the batch size of the data loader thus is set to 1 (default)
    # epoch size is how many elements the iterator of the generator will provide, NOTE should not be too small, because it have a significant overhead p=0.05
    training_set = CounterStrikeDataset(DATASET_CONFIG,
                                        TRAIN_CONFIG,
                                        dataset_files_partition,
                                        isValidationSet=False)
    training_generator = torch.utils.data.DataLoader(training_set,
                                                     batch_size=1,
                                                     shuffle=False)

    validation_set = CounterStrikeDataset(DATASET_CONFIG,
                                          TRAIN_CONFIG,
                                          dataset_files_partition,
                                          isValidationSet=True)
    validation_generator = torch.utils.data.DataLoader(validation_set,
                                                       batch_size=1,
                                                       shuffle=True)

    print("(Mini-)batches per epoch: " + str(training_set.__len__()))

    model = models.SharedWeightsCSGO(
        num_all_player_features=training_set.num_all_player_features,
        shared_layer_sizes=TRAIN_CONFIG["topography"]["shared_layers"],
        dense_layer_sizes=TRAIN_CONFIG["topography"]["dense_layers"])

    model.to(device)

    print("Creating tensorboard summary writer")

    writer = SummaryWriter(str(Path(PATH_RESULTS / run_name)))
    writer.add_hparams(
        {
            "feature_set": TRAIN_CONFIG["training"]["feature_set"],
            "lr": TRAIN_CONFIG["training"]["lr"],
            "batch_size": training_set.batch_row_size
        }, {})

    #dummy_X, dummy_y, dummy_player_i = next(iter(training_generator))
    #dummy_X = [(player_X[0, :]).to(device) for player_X in dummy_X]
    #writer.add_graph(model, dummy_X)

    #criterion = nn.CrossEntropyLoss()
    binary_classification_loss = torch.nn.BCELoss()
    #binary_classification_loss = torch.nn.BCEWithLogitsLoss()
    '''if (loss_calculation_mode == 'weighted'):
        #TODO: Calc weight for loss function
        binary_classification_loss = torch.nn.BCELoss()'''

    optimizer = OptimizerType(model.parameters(),
                              lr=TRAIN_CONFIG["training"]["lr"])

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

    #
    ''' PROGRESS BARS '''
    epoch_display_stats = {
        "loss": 100.0,
        "accuracy": 0.0,
        "die_accuracy": 0.0,
        "not_die_accuracy": 0.0
    }

    train_prog_bar = tqdm(total=training_set.num_epoch,
                          desc="EPOCH 0",
                          dynamic_ncols=True)
    validation_prog_bar = tqdm(desc="Validation epoch",
                               total=len(validation_generator),
                               dynamic_ncols=True)
    epoch_prog_bar = tqdm(total=training_set.num_batches,
                          desc="Epoch progress",
                          postfix=epoch_display_stats,
                          dynamic_ncols=True)

    for epoch_i in range(training_set.num_epoch):

        now = time.time()
        train_prog_bar.set_description(f"EPOCH {epoch_i}")

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
            X = [(player_X[0, :]).to(device) for player_X in X]
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
            #TODO: Maybe nn.BCEWithLogitsLoss?
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
            epoch_loss_target_player.append(
                batch_loss_target_player.cpu().detach().numpy().astype(
                    np.float32))

            batch_loss_all_player = binary_classification_loss(
                output, y).cpu().detach().numpy().astype(np.float32)
            epoch_loss_all_player.append(batch_loss_all_player)

            #
            '''
                NOTE: Log accuracy values 
            '''

            # Acc values for all players and for targeted player
            batch_accuracy_all_player = ((output > 0.5) == (
                y > 0.5)).cpu().numpy().astype(np.float32)
            batch_accuracy_target_player = ((player_i_output > 0.5) == (
                y[:, player_i] > 0.5)).cpu().numpy().reshape(-1).astype(
                    np.float32)

            batch_accuracy_all_player_mean = batch_accuracy_all_player.mean()
            batch_accuracy_target_player_mean = batch_accuracy_target_player.mean(
            )

            # Accuracy for predicting deaths correctly and survival
            batch_accuracy_die = ((output > 0.5) == (y > 0.5)).view(-1)[
                y.view(-1) > 0.5].cpu().numpy().reshape(-1).astype(np.float32)
            batch_accuracy_not_die = ((output > 0.5) == (y > 0.5)).view(-1)[
                y.view(-1) < 0.5].cpu().numpy().reshape(-1).astype(np.float32)

            epoch_accuracy_all_player.append(
                batch_accuracy_all_player_mean
            )  # Add mean accuracy to epoch accuracies
            epoch_accuracy_target_player.append(
                batch_accuracy_target_player_mean)

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

        #TODO: Add accuracies for predictions per time window [1,2,3....,20]

        epoch_prog_bar.reset()

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

        all_train_losses.append(np.array(epoch_loss_all_player).mean())
        all_train_accuracies.append(np.array(epoch_accuracy_all_player).mean())
        all_train_target_accuracies.append(
            np.array(epoch_accuracy_target_player).mean())
        all_train_die_notdie_accuracies.append(
            (np.array(epoch_accuracy_die).mean(),
             np.array(epoch_accuracy_not_die).mean()))
        '''
        TODO: For continuuos death labels

        for timeslot_i in range(20):
            all_train_per_sec_accuracies[timeslot_i].append(
                np.array(epoch_per_sec_accuracies[timeslot_i]).mean())
            all_train_per_sec_predictions[timeslot_i].append(
                np.array(epoch_per_sec_predictions[timeslot_i]).mean())
            all_train_per_sec_predictions_std[timeslot_i].append(
                np.array(epoch_per_sec_predictions[timeslot_i]).std())
        '''

        #
        '''
            TODO: Stop when training get too slow
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
                for val_batch_i, (X, y,
                                  player_i) in enumerate(validation_generator):
                    X = [(hero_X[0, :]).to(device) for hero_X in X]
                    y = (y[0, :]).to(device)

                    output = model(X)
                    output = torch.sigmoid(output)
                    output_np = output.cpu().detach().numpy()

                    batch_loss = binary_classification_loss(
                        output, y).cpu().detach().numpy().reshape(-1).mean()
                    epoch_loss_all_player.append(batch_loss)

                    accuracy_vec = ((output > 0.5) == (
                        y > 0.5)).cpu().numpy().reshape(-1).astype(np.float32)
                    epoch_accuracy_all_player.append(accuracy_vec.mean())

                    epoch_all_pred.extend(output_np.reshape(-1))
                    epoch_all_y.extend(y.cpu().numpy().reshape(-1))

                    validation_prog_bar.update()

                    if (val_batch_i >= log_every_num_batches) and (
                            val_batch_i % log_every_num_batches) == 0:

                        validation_prog_bar.set_postfix({
                            "val. loss":
                            np.array(
                                epoch_loss_all_player[-log_every_num_batches:]
                            ).mean(),
                            "val. accuracy":
                            np.array(epoch_accuracy_all_player[
                                -log_every_num_batches:]).mean()
                        })

            writer.add_scalars(
                "Validation/Loss",
                {'Loss all players': np.array(epoch_loss_all_player).mean()},
                epoch_i)

            epoch_accuracy_all_player_mean = np.array(
                epoch_accuracy_all_player).mean()

            writer.add_scalars(
                "Validation/Accuracy",
                {"Accuracy all players": epoch_accuracy_all_player_mean},
                epoch_i)
            '''writer.add_scalar(
                "Validation/D", {
                    "":
                    np.array(epoch_accuracy_die).mean(),
                    "Accuracy for Survival (not die)":
                    np.array(epoch_accuracy_not_die).mean()
                }, epoch_i)'''

            validation_roc_auc_score = roc_auc_score(epoch_all_y,
                                                     epoch_all_pred)
            validation_average_precision_score = average_precision_score(
                epoch_all_y, epoch_all_pred)

            writer.add_scalars(
                "Validation/Scores", {
                    "ROC Score": validation_roc_auc_score,
                    "Average Precision Score":
                    validation_average_precision_score
                }, epoch_i)

            all_validation_roc_scores.append(validation_roc_auc_score)
            all_validation_pr_scores.append(validation_average_precision_score)

            all_validation_losses.append(
                np.array(epoch_loss_all_player).mean())

            all_validation_accuracies.append(epoch_accuracy_all_player_mean)

            validation_prog_bar.write(
                f"Validation mean accuracy: {epoch_accuracy_all_player_mean}")

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
            f"########################## \nEpoch done {epoch_i} | loss: {np.array(epoch_loss_all_player).mean()} | target accuracy: {np.array(epoch_accuracy_target_player).mean()}"
        )
        train_prog_bar.write(
            f"Epoch took: {time.time() - now} \n ==========================================================="
        )
        sys.stdout.flush()

        #
        '''
        if (epoch_i % 10) == 9:
            np.save('epoch_per_sec_predictions.npy',
                    np.array(epoch_per_sec_predictions))

        if (epoch_i % 10) == 9:
            np.save('all_train_per_sec_predictions.npy',
                    np.array(all_train_per_sec_predictions))
            np.save('all_train_per_sec_predictions_std.npy',
                    np.array(all_train_per_sec_predictions_std))
        '''

        model_name = data_loader.MODEL_NAME_TEMPLATE.format(
            run_name=run_name,
            epoch_i=epoch_i,
            feature_set=training_set.feature_set)

        if (epoch_i % TRAIN_CONFIG["training"]["checkpoint_epoch"]
            ) == 0 and epoch_i > 0:
            torch.save(model.state_dict(), f'models/{model_name}.model')

        #CLI
        train_prog_bar.update()

    writer.close()


if __name__ == "__main__":
    parser = ap.ArgumentParser(
        description="Training script for encounter prediction network")

    parser.add_argument("-dataconf",
                        default='config/dataset_config.json',
                        help="Config of dataset")

    parser.add_argument("-trainconf",
                        default='config/train_config.json',
                        help="Config for training")
    parser.add_argument(
        "-name",
        default=f'run_{date.today().strftime("%S-%M-%H-%d-%b-%Y")}',
        help="Name of training run")
    '''parser.add_argument(
        "-loss_mode",
        choices=['all', 'target', 'weighted'],
        default='target',
        help=
        "WIP! ---- How calculate loss? For all outputs equally, only for the target player of the batch, or weighted for deaths vs non deaths labels."
    )'''
    parser.add_argument("-verbose", default=2)

    args = parser.parse_args()

    train_csgo(dataset_config_path=args.dataconf,
               train_config_path=args.trainconf,
               run_name=args.name,
               loss_calculation_mode=args.loss_mode)
