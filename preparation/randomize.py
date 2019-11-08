import numpy as np
import pandas as pd

import sys
import os
import time
from pathlib import Path
from typing import List

from tqdm import tqdm

sys.path.insert(0, str(Path.cwd() / 'preparation/'))

import data_loader


def randomize_processed_file(file_path: Path):
    df = data_loader.load_h5_as_df(file_path, True, 'player')

    return randomize_processed_dataframe(df)


def randomize_processed_dataframe(df: pd.DataFrame):
    '''
    NOTE: Not used right now
    '''
    return df.sample(frac=1)


def randomize_processed_files(files_list: List[Path],
                              randomized_files_path: Path,
                              chunk_row_size: int = 4096,
                              worker: int = 0):
    '''
    Combines multiple .h5 together and shuffles them. After that they are split into equal sized chunks and saved. 
    Leftover is also saved as a leftover file
    '''

    leftover_df: pd.DataFrame = None
    if not os.path.exists(str(randomized_files_path)):
        os.makedirs(str(randomized_files_path))
    else:
        try:
            leftover_file = data_loader.get_files_in_directory(
                randomized_files_path, '.leftover')[0]
            leftover_df = data_loader.load_h5_as_df(leftover_file, True)
        except:
            pass

    # Ticks are dropped, because they are not needed anymore
    df = data_loader.load_h5_as_df(files_list[0], True)

    if len(files_list) > 1:
        for h5_file in files_list[1:]:
            new_df = data_loader.load_h5_as_df(h5_file, True)
            df = pd.concat([df, new_df])

    df = df.sample(frac=1)

    last_chunk_df: pd.DataFrame
    df_length = len(df)

    # Split shuffled data into smaller chunks
    for counter, start_index in enumerate(range(0, df_length, chunk_row_size)):
        end_index = min(df_length, start_index + chunk_row_size)
        last_chunk_df = df.iloc[start_index:end_index]

        normal_chunk_file_name = f"data_chunk_{worker}_{counter}.h5"

        if (len(last_chunk_df) >= chunk_row_size):
            '''
            If enough data for a full chunk
            '''
            last_chunk_df.to_hdf(str(randomized_files_path /
                                     normal_chunk_file_name),
                                 key='player_info',
                                 mode='w')
        else:
            if (leftover_df is not None):
                last_chunk_df = pd.concat([leftover_df, last_chunk_df])

            if (len(last_chunk_df) > chunk_row_size):
                # Save new chunk as normal chunk, cutoff at length of normal chunk. Rest is leftover
                (last_chunk_df.iloc[:min(len(last_chunk_df), chunk_row_size)]
                 ).to_hdf(str(randomized_files_path / normal_chunk_file_name),
                          key='player_info',
                          mode='w')
                # Save rest as leftover
                (last_chunk_df.iloc[chunk_row_size:]).to_hdf(str(
                    randomized_files_path / f"leftover.h5.leftover"),
                                                             key='player_info',
                                                             mode='w')
            else:
                # Save last chunk as leftover
                last_chunk_df.to_hdf(str(randomized_files_path /
                                         f"leftover.h5.leftover"),
                                     key='player_info',
                                     mode='w')


if __name__ == "__main__":
    #randomize_processed_files(["parsed_files/vitality-vs-sprout-m1-overpass.h5"],Path("./randomized_files/"))
    pass