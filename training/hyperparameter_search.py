import json
import os
import random
import subprocess
import sys
import platform
import secrets
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path.cwd() / 'preparation/'))

import data_loader


def generate_random_config(default_config, i):
    #np.random.seed()

    new_conf = default_config.copy()

    # choose a feature set
    new_conf["training"]["feature_set"] = random.choice(["training_all"])

    new_conf["training"]["label_set"] = "discrete_die_within_5_seconds"

    new_conf["topography"] = [
        {
            "shared_layers": [400, 200, 120, 40],
            "dense_layers": [300, 150, 75, 32]
        },  # 4,4
        {
            "shared_layers": [1024, 512, 256, 128, 64],
            "dense_layers": [512, 256, 128, 64, 32]
        },  # 5,5
        {
            "shared_layers": [2000, 1000, 500, 250, 125, 62],
            "dense_layers": [512, 256, 128, 64, 32]
        },  # 6,5
        {
            "shared_layers": [1024, 512, 256, 128, 64, 32],
            "dense_layers": [300, 150, 75]
        },  # 6,2
        {
            "shared_layers": [256, 128, 64],
            "dense_layers": [1024, 512, 256, 128, 64, 32]
        },  # 3,6
        {
            "shared_layers": [400, 200, 80],
            "dense_layers": [1000, 500, 250, 100, 50]
        },  # 3,5
        {"shared_layers" : [256, 128, 64, 32],"dense_layers" : [2048, 1024, 512, 256, 128]},            # 2,1
        #{"shared_layers" : [100,60,20],"dense_layers" : [500, 250, 100, 50]},        # 3,1
        #{"shared_layers" : [200,100,60,20],"dense_layers" : [100]},    # 4,1
        #{"shared_layers" : [100,60,20],"dense_layers" : [150,75]},     # 3,2
        {"shared_layers" : [200,100,60,20],"dense_layers" : [256, 128, 64]} # 4,2
        #{"shared_layers": [200, 100, 40],"dense_layers": [300, 150, 75]}  # 3,3
    ][i]

    np.random.seed()

    # choose a batch size
    new_conf["training"]["batch_size"] = 128

    #new_conf["training"]["num_epoch"] = 100

    # choose optimitzer
    #new_conf["optimizer"] = "Adam"  # np.random.choice(["Adam","SGD"])
    new_conf["training"][
        "lr"] = secrets.choice([0.005, 0.002, 0.001, 0.0005, 0.0001, 0.00005,0.00001])
    #new_conf["training"]["lr"] = {"lr": 10**np.random.uniform(-1, -5)}

    #new_conf["validation_epoch_size"] = 3

    #new_conf["use_gpu"] = "true"  # if available

    #if ENVIRONMENT == "slurm_debug": new_conf["log_at_every_x_sample"] = 7680

    return new_conf


def get_configs(config, n):
    cfgs = []
    for i in range(n):
        cfgs.append(generate_random_config(config, i))
    return cfgs


def run_experiments(experiment_name, n):
    shell_bool = False
    if (platform.system() == 'Windows'): shell_bool = True

    configs = get_configs(
        data_loader.load_json(Path('config/train_config.json')), n)

    cwd = Path.cwd()

    for i in range(n):
        try:
            os.mkdir(str(cwd / f'config/{experiment_name}'))
        except:
            pass

        with open(
                str(cwd /
                    f'config/{experiment_name}/train_config_{experiment_name}_{i}.json'
                    ), 'w') as outfile:
            json.dump(configs[i], outfile)

        # Use parsing.js to parse demo FIXME: Depending on system it may be 'node' or 'nodejs'
        completedProcess = subprocess.run([
            'python3', 'training/csgotest.py', '-name',
            f'{experiment_name}_{i}', '-trainconf',
            str(cwd /
                f'config/{experiment_name}/train_config_{experiment_name}_{i}.json'
                )
        ],
                                          shell=shell_bool)


if __name__ == "__main__":
    run_experiments('third_exp', 8)
