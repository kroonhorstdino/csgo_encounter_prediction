import os
import numpy as np
import pandas as pd

# import preprocess
# import data_loader

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd() / 'preparation/'))


class SharedWeightsCSGO(torch.nn.Module):
    def __init__(self,
                 num_all_player_features,
                 num_players=10,
                 num_labels=10,
                 shared_layer_sizes=None,
                 dense_layer_sizes=None):
        super(SharedWeightsCSGO, self).__init__()

        if shared_layer_sizes is None:
            shared_layer_sizes = [200, 100, 60, 20]
        if dense_layer_sizes is None:
            dense_layer_sizes = [150, 75]

        # have to use ModuleList because using a plain list fails to populate model.parameters()
        self.shared_layers = nn.ModuleList([])
        self.dense_layers = nn.ModuleList([])

        #
        '''
        Shared Weight Layers
        '''

        # First layer of shared layers
        previous_layer_size = int(num_all_player_features / num_players)

        for layer_size in shared_layer_sizes:
            self.shared_layers.append(
                torch.nn.Linear(in_features=previous_layer_size,
                                out_features=layer_size))
            previous_layer_size = layer_size

        #
        ''' 
        Dense FF layers
        '''

        # this is the size after the concatenation
        previous_layer_size = num_players * previous_layer_size

        for layer_size in dense_layer_sizes:
            self.dense_layers.append(
                torch.nn.Linear(in_features=previous_layer_size,
                                out_features=layer_size))
            previous_layer_size = layer_size

        # Add last layer

        dense_output_layer = torch.nn.Linear(in_features=previous_layer_size,
                                             out_features=num_labels)
        self.dense_layers.append(dense_output_layer)

    def forward(self, all_player_features):
        all_outputs = []
        for player_x in all_player_features:  # Go through all shared networks with each player's features
            for shared_layer in self.shared_layers:  # Go through all layers in shared network
                player_x = torch.relu(shared_layer(player_x))
            all_outputs.append(player_x)

        # Concatenate output of all shared networks
        x = torch.cat(all_outputs, dim=1)

        # Go through dense layers
        for layer_i, final_layer in enumerate(self.dense_layers):
            x = final_layer(x)
            if layer_i < (len(self.dense_layers) -
                          1):  # no ReLU for the last layer
                x = torch.relu(x)

        return x