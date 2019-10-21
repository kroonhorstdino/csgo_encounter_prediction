import os

import numpy as np
import typing


# TODO: Hyperparameter search
def search_hyperparameters(config: dict) -> dict:
    new_config: dict = {}

    new_config["training"]["batch_size"] = np.random.choice(
        [4, 8, 16, 32, 64, 128, 256])
    new_config["training"]["lr"] = [
        0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005
    ]

    if (len(config) > 0): new_config = config.update(new_config)
    return new_config


if __name__ == "__main__":
    print(search_hyperparameters({}))
