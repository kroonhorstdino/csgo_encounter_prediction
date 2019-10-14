import numpy as np
import pandas as pd

import sys
import os
from pathlib import Path
from typing import List

import data_loader


def randomize_processed_files(fileList: List[Path], size_of_chunks_in_rows=5000, max_num_of_files: int = None):
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
            new_df = pd.read_hdf(file)
            df = pd.concat(df, )

    df = df.sample(frac=1)  # Shuffling #TODO maybe sklearn shuffle?

    df_length = len(df)

    last_chunk_df = None

    # Split dataframe into roughly equal parts (based on row count) and then save them to directory
    for i in range(0, df_length, size_of_chunks_in_rows):
        last_chunk_df = df[i: min(df_length - 1, i + size_of_chunks_in_rows)]

    df.to_hdf(str(Path.cwd() // 'parsed_files' // 'data.h5'),
              key='df', mode='w')  # TODO Split into chunks!
