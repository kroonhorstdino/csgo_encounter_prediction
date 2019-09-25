from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from pydoc import locate
import commentjson
from termcolor import colored
import models
import os
import random
import glob
import itertools
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import matplotlib.animation as animation

# in case it is called from a different location
sys.path.append(os.path.dirname(os.path.realpath(__file__)))


def train_pytorch():
    '''
    # is there a config in the current directory?
    config_path = "config.json"
    if not os.path.isfile("config.json"):
        # use default config
        config_path = os.path.dirname(
            os.path.realpath(__file__)) + "/config/default.json"

    with open(config_path) as f:
        config = commentjson.load(f)

    import pprint
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(config)
    sys.stdout.flush()

    WHO_DIES_NEXT_MODE = config["predict_who_dies_next"]'''

    use_cuda = True and torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    print("using device: ", device)

    '''model_type = locate(config["model"])
    get_feature_indicies_fn = locate(config["feature_set"])
    get_label_indicies_fn = locate(config["lable_set"])'''

    batch_size = 100  # config["batch_size"]
    print(type(batch_size))
    # print(type(config["log_at_every_x_sample"]))
    epoch_size = 1  # int(config["log_at_every_x_sample"] / batch_size)
    print("epoch_size: ", epoch_size)
    '''checkpoint_frequency = int(
        config["chekpoint_at_every_x_sample"] / (epoch_size * batch_size))
    validation_epoch_sice = config["validation_epoch_size"]'''

    OptimizerType = torch.optim.Adam

    # YARCC
    # trainingDataFiles = glob.glob("/scratch/ak1774/data/train/*.h5")
    # validationDataFiles = glob.glob("/scratch/ak1774/data/validation/*.h5")

    # Viking
    # glob.glob("/mnt/lustre/groups/cs-dclabs-2019/esport/death_prediction_data/randomized_data/train/*.h5")
    trainingDataFiles = glob.glob(
        str(Path.cwd() / 'randomized_data' / 'train') + '/*.h5')
    # glob.glob("/mnt/lustre/groups/cs-dclabs-2019/esport/death_prediction_data/randomized_data/validation/*.h5")
    validationDataFiles = glob.glob(
        str(Path.cwd() / 'randomized_data' / 'validation') + '/*.h5')

    '''
    # trainingDataFiles = glob.glob("/scratch/staff/ak1774/shared_folder/data/train/*.h5")
    # validationDataFiles = glob.glob("/scratch/staff/ak1774/shared_folder/data/validation/*.h5")

    example_data = data_loader.load_data_from_file(trainingDataFiles[0])
    hero_feature_indicies = get_feature_indicies_fn(example_data)

    if WHO_DIES_NEXT_MODE == True:
        label_indicies = get_label_indicies_fn(example_data)
    else:
        # data = data_loader.load_data_from_file(np.random.choice(np.array(self.file_list)))# random file in here
        label_indicies = get_label_indicies_fn(example_data) ,config["label_set_arg"])

    inputFeatureSize = len(hero_feature_indicies[0])
    outputFeatureSize = len(label_indicies)

    if WHO_DIES_NEXT_MODE == True and outputFeatureSize != 11:
        print("error, bad config, label set and prediction mode mismatch")
        raise "error, bad config, label set and prediction mode mismatch"
    elif WHO_DIES_NEXT_MODE == False and outputFeatureSize != 10:
        print("error, bad config, label set and prediction mode mismatch")
        raise "error, bad config, label set and prediction mode mismatch"
    '''

    # the dataset returns a batch when called (because we get the whole batch from one file), the batch size of the data loader thus is set to 1 (default)
    # epoch size is how many elements the iterator of the generator will provide, NOTE should not be too small, because it have a significant overhead p=0.05
    training_set = CounterStrikeDataset(
        (str(Path.cwd() / 'parsed_files' / 'positions.csv')))
    training_generator = torch.utils.data.DataLoader(
        training_set, shuffle=True)

    print(training_set.df.iloc[200])
    print(str(training_set.df.index.size))

    '''validation_set = DotaDataset(file_list=validationDataFiles, batch_size=batch_size, epoch_size=validation_epoch_sice,
                                 feature_indicies=hero_feature_indicies, label_indicies=label_indicies, who_dies_next_mode=WHO_DIES_NEXT_MODE, is_validation=False)  # actually we want the same distribution, so we can compare loss, so dont do anything differently in case of validation
    validation_generator = torch.utils.data.DataLoader(
        validation_set, num_workers=20, worker_init_fn=worker_init_fn)'''

    # model = models.SimpleFF(inputFeatureSize,outputFeatureSize)
    '''model = model_type(inputFeatureSize, outputFeatureSize,
                       **config["model_params"])
    '''
    model = models.SimpleFF(162, 10)

    model.to(device)
    # print(model.final_layers)

    criterion = nn.CrossEntropyLoss()
    binary_classification_loss = torch.nn.BCELoss()
    # optimizer = OptimizerType(model.parameters(), **config["optimizer_params"])
    optimizer = OptimizerType(model.parameters(), lr=0.001)

    '''if True:

        all_train_losses = []
        all_train_accuracies = []
        all_train_kill_nokill_accuracies = []
        all_train_per_second_accuracies = []

        all_validation_losses = []
        all_validation_accuracies = []
        all_validation_kill_nokill_accuracies = []
        all_validation_per_second_accuracies = []

        for epoch_i in range(epoch_size):

            now = time.time()

            np.random.seed()  # reset seed   https://github.com/pytorch/pytorch/issues/5059  data loader returns the same values

            epoch_losses = []
            epoch_overall_accuracies = []
            epoch_kill_accuracies = []
            epoch_no_kill_accuracies = []
            epoch_one_sec_accuracies = []
            epoch_two_sec_accuracies = []
            epoch_three_sec_accuracies = []
            epoch_four_sec_accuracies = []
            epoch_five_sec_accuracies = []

            for sub_epoch_i, (X, y) in enumerate(training_generator):

                # since we get a batch of size 1 of batch of real batch size, we take the 0th element
                # X = [(hero_X[0, :]).to(device) for hero_X in X]
                # y = torch.argmax(y[0, :], dim=1).to(device)
                # death_times = death_times[0]

                # Forward + Backward + Optimize
                optimizer.zero_grad()
                output = model(X)

                y = torch.rand(-1)
                # Compares labels from sample with actual output
                loss = criterion(output, y)
                accuracy_vec = (torch.argmax(output, 1) == y).cpu(
                ).numpy().reshape(-1).astype(np.float32)

                loss.backward()
                optimizer.step()

                epoch_losses.append(loss.cpu().detach().numpy().reshape(-1)[0])

                # (overall_accuracy, (kill_accuracy, no_kill_accuracy),
                # (one_sec_accuracy, two_sec_accuracy, three_sec_accuracy, four_sec_accuracy, five_sec_accuracy)) = calculate_detailed_accuracies(accuracy_vec, death_times, y)

                
                epoch_overall_accuracies.append(overall_accuracy)
                epoch_kill_accuracies.extend(kill_accuracy)
                epoch_no_kill_accuracies.extend(no_kill_accuracy)
                epoch_one_sec_accuracies.extend(one_sec_accuracy)
                epoch_two_sec_accuracies.extend(two_sec_accuracy)
                epoch_three_sec_accuracies.extend(three_sec_accuracy)
                epoch_four_sec_accuracies.extend(four_sec_accuracy)
                epoch_five_sec_accuracies.extend(five_sec_accuracy)

                if sub_epoch_i > 0 and (sub_epoch_i % 50) == 0:
                    print(epoch_i, " ", sub_epoch_i, " loss: ", np.array(epoch_losses[-49:]).mean(
                    ), " accuracy: ", np.array(epoch_overall_accuracies[(-49*y.shape[0]):]).mean())
                    sys.stdout.flush()
                

            all_train_losses.append(np.array(epoch_losses).mean())
            all_train_accuracies.append(
                np.array(epoch_overall_accuracies).mean())
            all_train_kill_nokill_accuracies.append(
                (np.array(epoch_kill_accuracies).mean(), np.array(epoch_no_kill_accuracies).mean()))
            all_train_per_second_accuracies.append((
                np.array(epoch_one_sec_accuracies).mean(),
                np.array(epoch_two_sec_accuracies).mean(),
                np.array(epoch_three_sec_accuracies).mean(),
                np.array(epoch_four_sec_accuracies).mean(),
                np.array(epoch_five_sec_accuracies).mean()
            ))

            # reset logs for validation
            epoch_losses = []
            epoch_overall_accuracies = []
            epoch_kill_accuracies = []
            epoch_no_kill_accuracies = []
            epoch_one_sec_accuracies = []
            epoch_two_sec_accuracies = []
            epoch_three_sec_accuracies = []
            epoch_four_sec_accuracies = []
            epoch_five_sec_accuracies = []

            # TODO Validation
            
            with torch.no_grad():
                for X, y, death_times in validation_generator:
                    X = [(hero_X[0, :]).to(device) for hero_X in X]
                    y = torch.argmax(y[0, :], dim=1).to(device)
                    death_times = death_times[0]

                    output = model(X)

                    loss = criterion(output, y)
                    accuracy_vec = (torch.argmax(output, 1) == y).cpu(
                    ).numpy().reshape(-1).astype(np.float32)

                    epoch_losses.append(
                        loss.cpu().detach().numpy().reshape(-1)[0])

                    (overall_accuracy, (kill_accuracy, no_kill_accuracy),
                     (one_sec_accuracy, two_sec_accuracy, three_sec_accuracy, four_sec_accuracy, five_sec_accuracy)) = calculate_detailed_accuracies(accuracy_vec, death_times, y)

                    epoch_overall_accuracies.append(overall_accuracy)
                    epoch_kill_accuracies.extend(kill_accuracy)
                    epoch_no_kill_accuracies.extend(no_kill_accuracy)
                    epoch_one_sec_accuracies.extend(one_sec_accuracy)
                    epoch_two_sec_accuracies.extend(two_sec_accuracy)
                    epoch_three_sec_accuracies.extend(three_sec_accuracy)
                    epoch_four_sec_accuracies.extend(four_sec_accuracy)
                    epoch_five_sec_accuracies.extend(five_sec_accuracy)

            all_validation_losses.append(np.array(epoch_losses).mean())
            all_validation_accuracies.append(
                np.array(epoch_overall_accuracies).mean())
            all_validation_kill_nokill_accuracies.append(
                (np.array(epoch_kill_accuracies).mean(), np.array(epoch_no_kill_accuracies).mean()))
            all_validation_per_second_accuracies.append((
                np.array(epoch_one_sec_accuracies).mean(),
                np.array(epoch_two_sec_accuracies).mean(),
                np.array(epoch_three_sec_accuracies).mean(),
                np.array(epoch_four_sec_accuracies).mean(),
                np.array(epoch_five_sec_accuracies).mean()
            ))

            # epoch over, checkpoint, report, check validation error
            print("Epoch done ", epoch_i, " loss: ", np.array(epoch_losses).mean(
            ), " accuracy: ", np.array(epoch_overall_accuracies).mean())

            # print("all_train_kill_nokill_accuracies ",len(all_train_kill_nokill_accuracies))
            PlotValues((all_train_losses, all_validation_losses),
                       "loss", ["train", "validation"])
            PlotValues((all_train_accuracies, all_validation_accuracies),
                       "accuracy", ["train", "validation"])

            PlotValues((*zip(*all_train_kill_nokill_accuracies), *zip(*all_validation_kill_nokill_accuracies)), "accuracy_kill",
                       ["train_kill", "train_no_kill", "validation_kill", "validation_no_kill"])

            sec_labels = ["1_sec", "2_sec", "3_sec", "4_sec", "5_sec"]
            PlotValues((*zip(*all_train_per_second_accuracies), *zip(*all_validation_per_second_accuracies)), "accuracy_sec",
                       [*["accuracy_train" + label for label in sec_labels], *["accuracy_validation" + label for label in sec_labels]])

            # np.save('losses.npy', np.array(mean_losses))
            # np.save('accuracies.npy', np.array(mean_accuracies))

            print("Epoch took: ", time.time()-now)
            sys.stdout.flush()

            # PlotValues(mean_validation_accuracies,"valid_accuracy")
            # PlotValues(mean_valid_overall_accuracies,"valid_overall_accuracy")

            # np.save('mean_valid_overall_accuracies.npy', np.array(mean_valid_overall_accuracies))
            # np.save('mean_validation_accuracies.npy', np.array(mean_validation_accuracies))

            if (epoch_i % 100) == 99:
                torch.save(model.state_dict(), "model" +
                           str(epoch_i) + ".model")
    '''

    for epoch_i in range(50000):

        now = time.time()

        np.random.seed()  # reset seed   https://github.com/pytorch/pytorch/issues/5059  data loader returns the same values

        epoch_overall_loss = []
        epoch_overall_accuracy = []
        epoch_target_accuracy = []
        epoch_die_accuracy = []
        epoch_not_die_accuracy = []
        epoch_per_sec_accuracies = [[] for _ in range(20)]
        epoch_per_sec_predictions = [[] for _ in range(20)]

        for sub_epoch_i, (X, y, death_times, player_i) in enumerate(training_generator):
            # since we get a batch of size 1 of batch of real batch size, we take the 0th element
            X = [(hero_X[0, :]).to(device) for hero_X in X]
            y = (y[0, :]).to(device)
            death_times = death_times[0]
            player_i = player_i[0].to(device)

            # Forward + Backward + Optimize
            optimizer.zero_grad()
            output = model(X)
            output = torch.sigmoid(output)
            output_np = output.cpu().detach().numpy()

            # only backpropagate the loss for player_i (so the training data is balanced)
            loss = binary_classification_loss(
                output[:, player_i], y[:, player_i])

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

            death_times = death_times.cpu().numpy()
            # death_times[death_times < 0] = 1000.0 # make invalid death times a large number

            for timeslot_i in range(19):
                mask_die_in_timeslot = np.logical_and(
                    (death_times > timeslot_i), (death_times < (timeslot_i+1)))
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

            if sub_epoch_i > 0 and (sub_epoch_i % 50) == 0:
                print(epoch_i, " ", sub_epoch_i, " loss: ", np.array(
                    epoch_overall_loss[-49:]).mean(), " accuracy: ", np.array(epoch_target_accuracy[-49:]).mean())
                # for timeslot_i in range(19):
                #    print("epoch_per_sec_predictions  ",len(epoch_per_sec_predictions[timeslot_i]))

                # print("die accuracy: ",np.array(epoch_die_accuracy[-49:]).mean())
                # print("not_die accuracy: ",np.array(epoch_not_die_accuracy[-49:]).mean())
                sys.stdout.flush()


train_pytorch()
