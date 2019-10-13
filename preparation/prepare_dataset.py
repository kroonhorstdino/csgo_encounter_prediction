import json
import os
import random
import subprocess
import sys
from pathlib import Path
import time
import argparse

import numpy as np
import pandas as pd

import data_loader
import preprocess

sys.path.append(Path.cwd().parent)
sys.path.append(Path.cwd())


def generate_dataset_file_partitions(files_path: Path, split_percentages=[0.5, 0.25, 0.25]):

    if(sum(split_percentages) != 1):
        print("Percentages do not add up to 100%. Now using >> 0.5 | 0.25 | 0.25 << split")
        split_percentages = [0.5, 0.25, 0.25]

    # Shuffle list of all .h5 files in the raw data directory
    raw_data_file_list = data_loader.get_files_in_dictionary(files_path, '.h5')
    random.shuffle(raw_data_file_list)

    return raw_data_file_list


def parse_data(demo_files_path: Path, parsed_csv_files_path: Path):

    print("Starting to parse demo files in folder: '" +
          str(demo_files_path) + "'")
    start_time_parse = time.process_time()

    # Get list of demo files in dictionary
    # Parse each demo file in demo_files_path
    files_paths = data_loader.get_files_in_dictionary(demo_files_path, '.dem')
    for file_path in files_paths:

        # Just to be sure
        csv_file = str(file_path)
        target_dir = str(parsed_csv_files_path)
        parsing_script_location = str(Path('./preparation/parsing.js'))

        # Use parsing.js to parse demo
        subprocess.run(['node', parsing_script_location,
                        csv_file, target_dir, str(args.verbose)], shell=True)

    # Time keeping
    end_time_parse = time.process_time()
    elapsed_time_parse = end_time_parse - start_time_parse

    print(f"Tried to parse >{len(files_paths)}< .dem files to .csv files")
    print("Parsing took >>" + str(elapsed_time_parse) + " seconds<<")


def preprocess_data(parsed_csv_files_path: Path, processed_files_path: Path):
    '''
    Processes data from matches in .csv files to .h5 files that contain all nessecary features for training

    Uses parameters in config #WIP
    '''

    parsed_csv_files_list = data_loader.get_files_in_dictionary(
        parsed_csv_files_path, '.csv')

    for parsed_csv_file in parsed_csv_files_list:
        df = data_loader.load_csv_as_df(parsed_csv_file)

        df = preprocess.add_die_within_sec_labels(df)
        df = preprocess.one_hot_encoding_weapons(df)

        target_path = str(processed_files_path / f'{parsed_csv_file.stem}.h5')

        df.to_hdf(target_path, key='df', mode='w')


def randomize_data():
    pass


def prepare_dataset():
    # PARSE DATA
    parse_data(Path(config["paths"]["demo_files_path"]),
               Path(config["paths"]["parsed_files_path"]))

    # PREPROCESS DATA
    preprocess_data(Path(config["paths"]["parsed_files_path"]),
                    Path(config["paths"]["processed_files_path"]))

    # RANDOMIZED DATA


if __name__ == '__main__':

    # preprocess_data(Path.cwd() / 'parsed_files', Path.cwd() / 'parsed_files')
    # Construct the argument parser
    ap = argparse.ArgumentParser(
        description="Script for preparing CSGO demo file dataset for machine learning \nConfig files must be present in './config/'")

    ap.add_argument("-config", type=str,
                    help="Path of config to use, default if not specified", required=False)
    ap.add_argument("-mode", "-m", required=False, choices=('all', 'parse', 'preprocess', 'randomize'), default='all', type=str,
                    help="Mode of preparation")
    # Option to delete old files that are generated in the in-between steps of preparation (.csv, .h5)
    ap.add_argument("-deleteold", required=False,
                    help="Verbosity intensity | Only 2 tiers")
    # Adjust verbosity level from 0 to 4 -v ... -vvvv
    ap.add_argument("-verbose", "-v", required=False, action='count',
                    help="Verbosity intensity | Only 2 tiers")

    args = ap.parse_args()

    start_time_all = time.process_time()

    # Load config
    config_path = 'config/prep_config.json'
    if (args.config):
        config_path = args.config

    config = None
    try:

        with open(config_path) as json_file:
            print("Loading config from " + config_path)
            config = json.load(json_file)
    except:
        print("Couldn't load config file, using default one")
        with open('config/prep_config.json') as json_file:
            config = json.load(json_file)

    if (args.mode != None):
        if (args.mode == 'all'):
            prepare_dataset()
        elif (args.mode == 'parse'):
            parse_data(config["paths"]["demo_files_path"],
                       config["paths"]["parsed_files_path"])
        elif (args.mode == 'preprocess'):
            preprocess_data(config["paths"]["parsed_files_path"],
                            config["paths"]["processed_files_path"])
        elif (args.mode == 'rand'):
            randomize_data()

    # Time keeping
    end_time_all = time.process_time()

    elapsed_time_all = end_time_all - start_time_all
    print("Preparation process in mode " +
          args.mode + "took >>" + str(elapsed_time_all) + " seconds<<")
