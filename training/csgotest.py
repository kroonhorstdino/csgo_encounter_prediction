from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from pydoc import locate
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

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import data_loader as data_loader
import preprocess as preprocess
import models as models

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset
import matplotlib.animation as animation

# in case it is called from a different location
sys.path.append(Path.cwd())
sys.path.append(Path.cwd() / 'preparation')

'''
class CounterStrikeDatasetSimple(Dataset):
    def __init__(self, filePath):
        print('read that thing')
        self.df = pd.read_csv(filePath, sep=',', na_values='-')
        self.df.fillna(0, inplace=True)

    def __getitem__(self, index):

        x = self.df.iloc[index]
        return x.values, torch.rand(1, 10), random.randrange(0, 10)

    def __len__(self):
        return self.df.index.size
'''


class CounterStrikeDataset(Dataset):
    def __init__(self, p_config, t_config, isValidationSet=False):
        if (not isValidationSet):
            self.dataset_files = data_loader.get_files_in_dictionary(p_config["paths"]["training_files_path"], '.h5')            
        else:
            self.dataset_files = data_loader.get_files_in_dictionary(p_config["paths"]["training_files_path"], '.h5')

        self.batch_row_size = t_config["batch_size"]

        self.num_chunks = len(self.dataset_files)
        # Num of rows for each chunk
        self.chunks_row_size = p_config["randomization"]["chunk_row_size"]
        # Num of batches in the entire dataset
        self.num_batches = int((self.num_chunks * self.chunks_row_size) / self.batch_row_size)
        # Amount of batches divided by num of files is num of batches per chunk
        self.batches_per_chunk = int(self.num_batches / self.num_chunks)

        self.num_epoch = t_config["num_epoch"]

        self.num_players = 10 #NOTE: Maybe there will be wingmen matches in the faaar future             
        self.death_time_window = p_config["preprocessing"]["death_time_window"]

        self.num_all_player_features = data_loader.get_num_player_features(data_loader.load_h5_as_df(self.dataset_files[0],False).columns)

    def __getitem__(self, index):

        player_i = random.randrange(0, 10)
        
        chosen_file = self.dataset_files[self.batch_index_to_chunk_index(index)]
        #start_index, end_index = self.get_indicies_in_chunk(index)
        chunk = data_loader.load_h5_as_df(chosen_file, True) 
        #.iloc[start_index:end_index] A batch is going to be loaded from this chunk >batches_per_chunk< times anyways. If random, may be not so bad NOTE: FIXME:

        player_features, classification_labels = data_loader.get_minibatch_balanced_player(chunk, player_i, batch_size=self.batch_row_size)

        return player_features, classification_labels, player_i

    def batch_index_to_chunk_index(self, batch_index : int) -> int:
        '''
            In what file/chunk is this batch
        '''
        return int(batch_index / self.batches_per_chunk)

    def get_indicies_in_chunk(self, batch_index) -> Tuple[int,int]:
        batch_in_chunk = batch_index % self.batches_per_chunk

        start_index_in_chunk = batch_in_chunk * self.batch_row_size
        end_index_in_chunk = start_index_in_chunk + self.batch_row_size #End index in selecting is exclusive in pandas
        
        return start_index_in_chunk, end_index_in_chunk 

    def __len__(self):
        return self.batches_per_chunk * self.num_chunks 

def train_csgo(prep_config_path : Path, train_config_path : Path):

    train_config = data_loader.load_config(train_config_path)
    prep_config = data_loader.load_config(prep_config_path)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    print("using device: ", device)

    OptimizerType = torch.optim.Adam

    # the dataset returns a batch when called (because we get the whole batch from one file), the batch size of the data loader thus is set to 1 (default)
    # epoch size is how many elements the iterator of the generator will provide, NOTE should not be too small, because it have a significant overhead p=0.05
    training_set = CounterStrikeDataset(prep_config, train_config, isValidationSet=False)
    training_generator = torch.utils.data.DataLoader(
        training_set, shuffle=True)

    print("Batches per epoch: " + str(training_set.__len__()))

    model = models.SharedWeightsCSGO(
        num_all_player_features=training_set.num_all_player_features , num_labels=10)

    model.to(device)

    criterion = nn.CrossEntropyLoss()
    binary_classification_loss = torch.nn.BCELoss()
    optimizer = OptimizerType(model.parameters(), lr=pow(3.06, -5))

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

    for epoch_i in range(training_set.num_epoch):

        now = time.time()

        np.random.seed()  # reset seed   https://github.com/pytorch/pytorch/issues/5059  data loader returns the same values

        epoch_overall_loss = []
        epoch_overall_accuracy = []
        epoch_target_accuracy = []
        epoch_die_accuracy = []
        epoch_not_die_accuracy = []
        epoch_per_sec_accuracies = [[] for _ in range(20)]
        epoch_per_sec_predictions = [[] for _ in range(20)]

        for batch_i, (X, y, player_i) in enumerate(training_generator):
            # since we get a batch of size 1 of batch of real batch size, we take the 0th element

            #death_times = death_times[0]

            # training_generator adds one dimension to each tensor
            X = [(hero_X[0, :]).to(device) for hero_X in X]
            y = (y[0, :]).to(device)
            player_i = player_i[0].to(device)

            # Forward + Backward + Optimize
            # Remove gradients from last iteration
            optimizer.zero_grad()

            # print(X)

            output = model(X)
            output = torch.sigmoid(output)
            output_np = output.cpu().detach().numpy()

            # only backpropagate the loss for player_i (so the training data is balanced)
            player_i_output = output[:, player_i]
            player_i_labels = y[:, player_i]

            loss = binary_classification_loss(player_i_output, player_i_labels)

            loss.backward()
            optimizer.step()

            overall_loss = binary_classification_loss(
                output, y).cpu().detach().numpy()
            epoch_overall_loss.append(overall_loss.reshape(-1).mean())
            accuracy_values = ((output > 0.5) == (
                y > 0.5)).cpu().numpy().astype(np.float32)

            target_accuracy = ((output[:, player_i] > 0.5) == (
                y[:, player_i] > 0.5)).cpu().numpy().reshape(-1).astype(np.float32)

            die_accuracy_vec = ((output > 0.5) == (
                y > 0.5)).view(-1)[y.view(-1) > 0.5].cpu().numpy().reshape(-1).astype(np.float32)
            not_die_accuracy_vec = ((output > 0.5) == (
                y > 0.5)).view(-1)[y.view(-1) < 0.5].cpu().numpy().reshape(-1).astype(np.float32)

            epoch_overall_accuracy.append(
                accuracy_values.reshape(-1).mean())
            epoch_target_accuracy.append(target_accuracy.mean())

            # these have varying size, so calculating the proper mean across batches takes more work
            epoch_die_accuracy.extend(die_accuracy_vec)
            epoch_not_die_accuracy.extend(not_die_accuracy_vec)

            # TODO #death_times = death_times.cpu().numpy()
            # death_times[death_times < 0] = 1000.0 # make invalid death times a large number

            '''for timeslot_i in range(19):
                    mask_die_in_timeslot = np.logical_and((death_times > timeslot_i), (death_times < (timeslot_i+1)))
                    epoch_per_sec_accuracies[timeslot_i].extend(
                        accuracy_values[mask_die_in_timeslot].reshape(-1))
                    epoch_per_sec_predictions[timeslot_i].extend(
                        output_np[mask_die_in_timeslot].reshape(-1))

                # and the rest
                mask_die_in_timeslot = (death_times > 19)
                epoch_per_sec_accuracies[19].extend(
                    accuracy_values[mask_die_in_timeslot].reshape(-1))
                epoch_per_sec_predictions[19].extend(
                    output_np[mask_die_in_timeslot].reshape(-1))
                '''

            if batch_i > 0 and (batch_i % 50) == 0:
                print(epoch_i, " ", batch_i, " loss: ", np.array(
                    epoch_overall_loss[-49:]).mean(), " accuracy: ", np.array(epoch_target_accuracy[-49:]).mean())
                # for timeslot_i in range(19):
                #    print("epoch_per_sec_predictions  ",len(epoch_per_sec_predictions[timeslot_i]))

                # print("die accuracy: ",np.array(epoch_die_accuracy[-49:]).mean())
                # print("not_die accuracy: ",np.array(epoch_not_die_accuracy[-49:]).mean())
                sys.stdout.flush()

        if (epoch_i % 10) == 9:
            np.save('epoch_per_sec_predictions.npy',
                    np.array(epoch_per_sec_predictions))

        all_train_losses.append(np.array(epoch_overall_loss).mean())
        all_train_accuracies.append(
            np.array(epoch_overall_accuracy).mean())
        all_train_target_accuracies.append(
            np.array(epoch_target_accuracy).mean())
        all_train_die_notdie_accuracies.append(
            (np.array(die_accuracy_vec).mean(), np.array(not_die_accuracy_vec).mean()))

        for timeslot_i in range(20):
            all_train_per_sec_accuracies[timeslot_i].append(
                np.array(epoch_per_sec_accuracies[timeslot_i]).mean())
            all_train_per_sec_predictions[timeslot_i].append(
                np.array(epoch_per_sec_predictions[timeslot_i]).mean())
            all_train_per_sec_predictions_std[timeslot_i].append(
                np.array(epoch_per_sec_predictions[timeslot_i]).std())

        print("Epoch done ", epoch_i, " loss: ", np.array(epoch_overall_loss).mean(
        ), " accuracy: ", np.array(epoch_target_accuracy).mean())
        print("Epoch took: ", time.time()-now)
        sys.stdout.flush()

        if (epoch_i % 10) == 9:
            np.save('all_train_per_sec_predictions.npy',
                    np.array(all_train_per_sec_predictions))
            np.save('all_train_per_sec_predictions_std.npy',
                    np.array(all_train_per_sec_predictions_std))

        if (epoch_i % 100) == 99:
            torch.save(model.state_dict(), "model" + str(epoch_i) + ".model")


if __name__ == "__main__":
    train_csgo('config/prep_config.json','config/train_config.json')