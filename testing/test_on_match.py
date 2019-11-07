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
sys.path.insert(0, str(Path.cwd() / 'training/'))

import data_loader
import models
import preprocess
import csgotest
import models
import prepare_dataset

PATH_RESULTS = Path('results')

TRAIN_CONFIG_PATH = Path('config' / 'train_config.json')
DATASET_CONFIG_PATH = Path('config' / 'dataset_config.json')

FEATURES_INFO_PATH = Path('config' / 'features_info.json')


def test_on_match(csv_match_file: Path, model: models.SharedWeightsCSGO):

    print("Start training process...")

    #TODO: Do something
    #log_every_num_batches = 250 - (50 * min((4, args.verbose)))

    TRAIN_CONFIG = data_loader.load_json(TRAIN_CONFIG_PATH)
    DATASET_CONFIG = data_loader.load_json(DATASET_CONFIG_PATH)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    print("Using device: ", device)

    OptimizerType = torch.optim.Adam

    dataset_files_partition = prepare_dataset.get_dataset_partitions(
        DATASET_CONFIG["paths"]["training_files_path"], [0.8, 0.2, 0])

    # the dataset returns a batch when called (because we get the whole batch from one file), the batch size of the data loader thus is set to 1 (default)
    # epoch size is how many elements the iterator of the generator will provide, NOTE should not be too small, because it have a significant overhead p=0.05
    test_set = csgotest.CounterStrikeDataset(DATASET_CONFIG,
                                             TRAIN_CONFIG,
                                             dataset_files_partition,
                                             isValidationSet=False)
    test_generator = torch.utils.data.DataLoader(test_set,
                                                 batch_size=1,
                                                 shuffle=False)

    print("(Mini-)batches per epoch: " + str(test_set.__len__()))

    model.to(device)

    print("Creating tensorboard summary writer")

    writer = SummaryWriter(str(Path(PATH_RESULTS / run_name)))
    writer.add_hparams({"test_set": "dummy_value"}, {})

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

    return 0