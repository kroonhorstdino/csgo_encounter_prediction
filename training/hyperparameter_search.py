import os

import numpy as np
import typing


# TODO: Hyperparameter search
def search_hyperparameters(config: dict) -> dict:
    new_config: dict = {}

    new_config["batch_size"] = np.random.choice([4, 8, 16, 32, 64, 128, 256])

    if (len(config) > 0): new_config = config.update(new_config)
    return new_config


if __name__ == "__main__":
    print(search_hyperparameters({}))
