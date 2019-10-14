import numpy as np
import pandas as pd

import sys
import os
from pathlib import Path
from typing import List

import preparation.data_loader as data_loader


def randomize_processed_files(fileList: List[Path], randomized_files_path : Path, chunk_row_size=4096, max_num_of_files: int = None):
    '''
    Combines multiple .h5 together and shuffles them. After that they are split into roughly equal chunks
    '''

    if max_num_of_files == None:
        max_num_of_files = len(fileList)

    # size_of_files = sum(os.path.getsize(f) for f in os.listdir('.') if os.path.isfile(f))
    df = data_loader.load_h5_as_df(fileList[0], True)
    # Ticks are dropped, because they are not needed anymore

    if max_num_of_files > 0:
        for file in fileList[1:max_num_of_files]:
            new_df = data_loader.load_h5_as_df(file, True)
            df = pd.concat(df, new_df)

    df = df.sample(frac=1)  # Shuffling #TODO maybe sklearn shuffle?

    df_length = len(df)

    last_chunk_df = None
    # Split dataframe into roughly equal parts (based on row count) and then save them to directory
    for i in range(0, df_length, chunk_row_size):
        last_chunk_df = df[i: min(df_length - 1, i + chunk_row_size)]

    df.to_hdf(str(randomized_files_path / 'random.h5'), key='df', mode='w')  # TODO Split into chunks!
