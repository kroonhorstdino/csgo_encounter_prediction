import json
import os
import random
import subprocess
import sys
from pathlib import Path
import time
import argparse
import platform

import numpy as np
import pandas as pd

from typing import List
import multiprocessing as mp

sys.path.insert(0, str(Path.cwd() / 'preparation/'))

import data_loader
import preprocess
import randomize


def get_dataset_partitions(files_path: Path,
                           split_percentages=[0.5, 0.25, 0.25]):

    if (sum(split_percentages) != 1):
        print(
            "Percentages do not add up to 100%. Now using >> 0.5 | 0.25 | 0.25 << split"
        )
        split_percentages = [0.5, 0.25, 0.25]

    # Shuffle list of all .h5 files in the raw data directory
    raw_data_file_list = data_loader.get_files_in_directory(files_path, '.h5')
    random.shuffle(raw_data_file_list)

    return raw_data_file_list


def parse_data(demo_files_paths: List[Path],
               parsed_csv_files_path: Path,
               worker_id=0):

    num_failed_parse = 0

    print(
        f"Worker{worker_id}: Starting to parse demo files with verbosity {str(args.verbose)}"
    )
    start_time_parse = time.time()

    # Get list of demo files in dictionary
    # Parse each demo file in demo_files_path

    # Create target dir if not exists
    if not os.path.exists(str(parsed_csv_files_path)):
        os.makedirs(str(parsed_csv_files_path))

    for file_path in demo_files_paths:

        # Just to be sure
        parsing_script_location = str(Path('./preparation/parsing.js'))
        csv_file = str(file_path)
        target_dir = str(parsed_csv_files_path)

        #Platform dependant NOTE:
        shell_bool = False
        if (platform.system() == 'Windows'): shell_bool = True

        # Use parsing.js to parse demo FIXME: Depending on system it may be 'node' or 'nodejs'
        completedProcess = subprocess.run([
            'node', parsing_script_location, csv_file, target_dir,
            str(args.verbose)
        ],
                                          shell=shell_bool)

        if (completedProcess.returncode == 1):
            num_failed_parse += 1
            print(f"Worker{worker_id}: Failed parsing...")

    # Time keeping
    end_time_parse = time.time()
    elapsed_time_parse = end_time_parse - start_time_parse

    print(
        f"Worker{worker_id}: Tried to parse >{len(demo_files_paths)}< .dem files to .csv files"
    )
    print(f"Worker{worker_id}: Parsing took >>" + str(elapsed_time_parse) +
          " seconds<<")


def preprocess_data(parsed_csv_files_list: List[Path],
                    processed_files_path: Path,
                    delete_old_csv: bool = False):
    '''
        Processes data from matches in .csv files to .h5 files that contain all nessecary features for training
        Uses parameters in config #WIP
    '''

    print("Preprocessing....")

    # Create target dir if not exists
    if not os.path.exists(str(processed_files_path)):
        os.makedirs(str(processed_files_path))

    for parsed_csv_file in parsed_csv_files_list:
        df = data_loader.load_csv_as_df(parsed_csv_file)

        df = preprocess.add_die_within_sec_labels(df)
        df = preprocess.undersample_pure_not_die_ticks(
            df, removal_frac=0.1)  #NOTE: Doesnt work yet
        df = preprocess.one_hot_encoding_weapons(df)

        target_path = str(processed_files_path / f'{parsed_csv_file.stem}.h5')

        #print(target_path)

        try:
            #Save to hdf and if specified, remove old csv file
            df.to_hdf(target_path, key='df', mode='w')
        except:
            print("Couldn't save to dataframe... ")
            raise
        else:
            if (delete_old_csv): os.remove(str(parsed_csv_file))


def randomize_data(processed_h5_files_list: List[Path],
                   randomized_files_path: Path,
                   files_per_worker: int = 5,
                   delete_old_h5: bool = False):

    print("Attempting to randomize and split in chunks...")

    if not os.path.exists(str(randomized_files_path)):
        os.makedirs(str(randomized_files_path))

    # Go through list with files_per_worker steps
    for i, list_index in enumerate(
            range(0, len(processed_h5_files_list), files_per_worker)):
        files_list = processed_h5_files_list[list_index:min(
            list_index + files_per_worker, len(processed_h5_files_list))]

        randomize.randomize_processed_files(
            files_list, randomized_files_path,
            config["randomization"]["chunk_row_size"], i)

    print("Randomization finished...")


def prepare_dataset():

    demo_files_list = data_loader.get_files_in_directory(
        Path(config["paths"]["demo_files_path"]), '.dem')
    ''' NOTE: For multiprocessing
    num_workers = mp.cpu_count()
    files_per_worker = int(len(demo_files_list) / num_workers)
    pool = mp.Pool(num_workers)

    parse_data_args: list
    #Generate args for parsing
    for index, list_index in enumerate(
            range(0, files_per_worker, len(demo_files_list))):
        parse_data_args.append(
            #Append parts of list for each worker
            [
                demo_files_list[list_index:min(list_index, len(demo_files_list)
                                               )],
                Path(config["paths"]["parsed_files_path"]), index
            ])

    pool.map(parse_data, parse_data_args)
    '''

    # PARSE DATA
    parse_data(demo_files_list,
               Path(config["paths"]["parsed_files_path"]),
               worker_id=0)

    # PREPROCESS DATA
    parsed_csv_files_list = data_loader.get_files_in_directory(
        Path(config["paths"]["parsed_files_path"]), ".csv")
    preprocess_data(parsed_csv_files_list,
                    Path(config["paths"]["processed_files_path"]))

    # RANDOMIZED DATA
    processed_h5_files_list = data_loader.get_files_in_directory(
        Path(config["paths"]["processed_files_path"]), ".h5")
    randomize_data(processed_h5_files_list,
                   Path(config["paths"]["training_files_path"]))


if __name__ == '__main__':

    # preprocess_data(Path.cwd() / 'parsed_files', Path.cwd() / 'parsed_files')
    # Construct the argument parser
    ap = argparse.ArgumentParser(
        description=
        "Script for preparing CSGO demo file dataset for machine learning \nConfig files must be present in './config/'"
    )

    ap.add_argument("-config",
                    type=str,
                    help="Path of config to use, default if not specified",
                    required=False)
    ap.add_argument("-mode",
                    "-m",
                    required=True,
                    choices=('all', 'parse', 'preprocess', 'randomize'),
                    type=str,
                    help="Mode of preparation")
    # Option to delete old files that are generated in the in-between steps of preparation (.csv, .h5)
    ap.add_argument("-deleteold",
                    required=False,
                    help="Verbosity intensity | Only 2 tiers")
    # Adjust verbosity level from 0 to 4 -v ... -vvvv
    ap.add_argument("-verbose",
                    "-v",
                    required=False,
                    action='count',
                    help="Verbosity intensity | Only 2 tiers")

    args = ap.parse_args()

    if (args.verbose == None):
        args.verbose = 4

    start_time_all = time.time()

    # Load config
    config_path = 'config/dataset_config.json'
    if (args.config):
        config_path = args.config

    config = data_loader.load_config(config_path)

    if (args.mode != None):
        if (args.mode == 'all'):
            print("Preparing entire dataset...")
            prepare_dataset()
        elif (args.mode == 'parse'):
            print("Only parsing...")
            demo_files_list = data_loader.get_files_in_directory(
                Path(config["paths"]["demo_files_path"]), '.dem')
            parse_data(demo_files_list,
                       Path(config["paths"]["parsed_files_path"]))
        elif (args.mode == 'preprocess'):
            print("Only preprocessing...")
            parsed_csv_files_list = data_loader.get_files_in_directory(
                Path(config["paths"]["parsed_files_path"]), ".csv")
            preprocess_data(parsed_csv_files_list,
                            Path(config["paths"]["processed_files_path"]))
        elif (args.mode == 'randomize'):
            processed_h5_files_list = data_loader.get_files_in_directory(
                Path(config["paths"]["processed_files_path"]), ".h5")
            randomize_data(processed_h5_files_list,
                           Path(config["paths"]["training_files_path"]))

    # Time keeping TODO: Time keeping is erroneous
    end_time_all = time.time()

    elapsed_time_all = end_time_all - start_time_all
    print("Preparation process in mode " + args.mode + " took >>" +
          str(elapsed_time_all) + " seconds<<")
