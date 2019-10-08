import sys
import os
import random

import data_loader
import preprocess

import numpy as np
import pandas as pd

from pathlib import Path

sys.path.append(Path.cwd().parent)
sys.path.append(Path.cwd())

def generate_dataset_file_partitions(files_path : Path, split_percentages=[0.5,0.25,0.25]):
    
    if(sum(split_percentages) != 1):
        print("Percentages do not add up to 100%. Now using >> 0.5 | 0.25 | 0.25 << split")
        split_percentages=[0.5,0.25,0.25]

    # Shuffle list of all .h5 files in the raw data directory
    raw_data_file_list = random.shuffle(data_loader.get_files_in_dictionary(files_path,'.h5'))

    return raw_data_file_list


def preprocess_dataset(parsed_csv_files_path: Path, processed_files_path: Path, config=None):
    '''
    Processes data from matches in .csv files to .h5 files that contain all nessecary features for training

    Uses parameters in config #WIP
    '''

    parsed_csv_files_list = data_loader.get_files_in_dictionary(parsed_csv_files_path, '.csv')

    for parsed_csv_file in parsed_csv_files_list:
        new_df = data_loader.load_csv_as_df(parsed_csv_file)

        df = preprocess.add_die_within_sec_labels(new_df)

        df.to_hdf(str(processed_files_path / f'{parsed_csv_file.stem}.h5'),
              key='df', mode='w')

           

if __name__ == '__main__':
    preprocess_dataset(Path.cwd() / 'parsed_files', Path.cwd() / 'parsed_files')

    
        
        

