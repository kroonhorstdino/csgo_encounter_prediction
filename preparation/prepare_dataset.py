import json
import os
import random
import subprocess
import sys
from pathlib import Path
import time
import argparse
import platform

from tqdm import tqdm

import numpy as np
import pandas as pd

from typing import List
import multiprocessing as mp

sys.path.insert(0, str(Path.cwd() / 'preparation/'))

import data_loader
import preprocess
import randomize


def get_dataset_partitions(files_path: Path, split_percentages=[0.8, 0.1,
                                                                0.1]):

    if (sum(split_percentages) != 1):
        print(
            "Percentages do not add up to 100%. Now using >> 0.5 | 0.25 | 0.25 << split"
        )
        split_percentages = [0.5, 0.25, 0.25]

    # Shuffle list of all .feather files in the raw data directory
    raw_data_file_list = data_loader.get_files_in_directory(
        files_path, '.feather')
    random.shuffle(raw_data_file_list)

    num_train_files = int(len(raw_data_file_list) * split_percentages[0])
    num_validation_files = int(len(raw_data_file_list) * split_percentages[1])
    num_test_files = int(len(raw_data_file_list) * split_percentages[2])

    dataset_partition_list = [[], [], []]

    dataset_partition_list[2] = raw_data_file_list[0:num_test_files]
    dataset_partition_list[1] = raw_data_file_list[num_test_files:
                                                   num_test_files +
                                                   num_validation_files]
    dataset_partition_list[0] = raw_data_file_list[num_test_files +
                                                   num_validation_files:]

    return dataset_partition_list


def parse_data(demo_files_paths: List[Path],
               parsed_csv_files_path: Path,
               worker_id=0,
               override=False):

    num_failed_parse = 0
    num_skipped = 0

    parse_prog_bar = tqdm(desc="Parsing progress", total=len(demo_files_paths))

    parse_prog_bar.write(
        f"Worker{worker_id}: Starting to parse demo files with verbosity {str(args.verbose)}"
    )

    # Get list of demo files in dictionary
    # Parse each demo file in demo_files_path

    # Create target dir if not exists
    if not os.path.exists(str(parsed_csv_files_path)):
        os.makedirs(str(parsed_csv_files_path))

    for file_path in demo_files_paths:
        parse_prog_bar.set_postfix({
            "Current file": file_path.stem,
            "Failed at": num_failed_parse
        })

        # Just to be sure
        parsing_script_location = str(Path('./preparation/parsing.js'))
        demo_file = str(file_path)
        target_dir = str(parsed_csv_files_path)

        new_file_path = Path(target_dir) / file_path.with_suffix('.csv').name
        if (override == False and Path.is_file(new_file_path)):
            num_skipped += 1
            parse_prog_bar.update()
            parse_prog_bar.set_postfix({
                "Current file": file_path.stem,
                "Failed at": num_failed_parse,
                "Skipped": num_skipped
            })
            continue  #Skip this file if we don't want to override again

        #Platform dependant NOTE:
        shell_bool = False
        if (platform.system() == 'Windows'): shell_bool = True

        # Use parsing.js to parse demo FIXME: Depending on system it may be 'node' or 'nodejs'
        completedProcess = subprocess.run([
            'node', parsing_script_location, demo_file, target_dir,
            str(args.verbose)
        ],
                                          shell=shell_bool)

        if (completedProcess.returncode == 1):
            num_failed_parse += 1
            print(f"Worker{worker_id}: Failed parsing...")

        parse_prog_bar.update()
        parse_prog_bar.set_postfix({
            "Current file": file_path.stem,
            "Failed at": num_failed_parse,
            "Skipped": num_skipped
        })

    parse_prog_bar.set_description("Preprocessing finished")
    parse_prog_bar.write(
        f"Finished parsing all files. Failed at {num_failed_parse} files")


def preprocess_data(parsed_csv_files_list: List[Path],
                    processed_files_path: Path,
                    delete_old_csv: bool = False,
                    time_window_to_next_death: int = 5,
                    override=False):
    '''
        Processes data from matches in .csv files to .feather files that contain all nessecary features for training
        Uses parameters in config #WIP
    '''

    #print(parsed_csv_files_list)

    prep_prog_bar = tqdm(desc="Preprocessing progress",
                         total=len(parsed_csv_files_list))

    prep_prog_bar.write("Preprocessing....")

    # Create target dir if not exists
    if not os.path.exists(str(processed_files_path)):
        os.makedirs(str(processed_files_path))

    for parsed_csv_file in parsed_csv_files_list:
        prep_prog_bar.set_postfix({"Current file": parsed_csv_file.stem})

        parsed_death_csv_file = parsed_csv_file.with_name(
            parsed_csv_file.stem + "_deaths.csv")

        target_path_new = str(processed_files_path /
                              f'{parsed_csv_file.stem}.feather')

        if (override == False and Path.is_file(Path(target_path_new))):
            prep_prog_bar.update()
            continue  #Skip this file if we don't want to override again

        df: pd.DataFrame
        df_deaths: pd.DataFrame

        try:
            df = data_loader.load_csv_as_df(parsed_csv_file)
            df_deaths = data_loader.load_csv_as_df(parsed_death_csv_file,
                                                   False)
        except:
            print("FILE HAD NO COLUMNS!")
            continue

        prep_prog_bar.write("Adding 'dies within x seconds' labels for a " +
                            str(time_window_to_next_death) +
                            " second time window to dataframe...")

        df = preprocess.add_die_within_sec_labels(df, df_deaths)
        df = preprocess.undersample_pure_not_die_ticks(
            df, removal_frac=0.1)  #NOTE: Doesnt work yet

        prep_prog_bar.write("Adding one hot encoding for weapons")
        df = preprocess.add_one_hot_encoding_weapons(df)

        prep_prog_bar.write("Adding one hot encoding for player aim on enemy")
        df = preprocess.add_one_hot_encoding_angles(df)

        str_target_path = str(Path(target_path_new).resolve())
        prep_prog_bar.write(str_target_path)

        try:
            df.reset_index(inplace=True)
            df.to_feather(str_target_path)
        except:
            prep_prog_bar.write(">>>>>>>>>>> Couldn't save to as file!")
            raise
        finally:
            prep_prog_bar.update()

        df = None

    prep_prog_bar.write("Preprocessing finished")
    prep_prog_bar.set_description("Preprocessing finished")


def randomize_data(processed_h5_files_list: List[Path],
                   randomized_files_path: Path,
                   files_per_worker: int = 16,
                   delete_old_h5: bool = False):

    rand_progress_bar = tqdm(desc="Randomization progress:",
                             total=len(processed_h5_files_list))

    rand_progress_bar.write("Attempting to randomize and split in chunks...")

    if not os.path.exists(str(randomized_files_path)):
        os.makedirs(str(randomized_files_path))

    for map_name in FEATURES_INFO["maps"]:
        processed_h5_files_per_map = list(
            filter(lambda file: map_name + "." in file.name,
                   processed_h5_files_list))

        if len(processed_h5_files_per_map) == 0:
            continue

        map_files_dir = str(randomized_files_path / map_name)
        if (not os.path.exists(map_files_dir)):
            os.makedirs(map_files_dir)

        # Go through list with files_per_worker steps
        for i, list_index in enumerate(
                range(0, len(processed_h5_files_per_map), files_per_worker)):
            files_list = processed_h5_files_per_map[list_index:min(
                list_index +
                files_per_worker, len(processed_h5_files_per_map))]

            rand_progress_bar.set_postfix({"Last file": files_list[0].stem})

            randomize.randomize_processed_files(
                files_list, Path(map_files_dir),
                config["randomization"]["chunk_row_size"], i)

            rand_progress_bar.update(len(files_list))

    rand_progress_bar.write("Randomization finished...")
    rand_progress_bar.set_description("Randomization finished")


def prepare_dataset():

    all_progress_bar = tqdm(desc="Entire progress", total=3)
    all_progress_bar.set_postfix({"Status": "Parsing"})

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
               worker_id=0,
               override=args.override)

    all_progress_bar.update()
    all_progress_bar.set_postfix({"Status": "Preprocessing"})

    # PREPROCESS DATA
    parsed_csv_files_list = list(
        filter(
            lambda name: "death" not in name,
            data_loader.get_files_in_directory(
                Path(config["paths"]["parsed_files_path"]), ".csv")))
    preprocess_data(parsed_csv_files_list,
                    Path(config["paths"]["processed_files_path"]),
                    override=args.override)

    all_progress_bar.update()
    all_progress_bar.set_postfix({"Status": "Randomization"})

    # RANDOMIZED DATA
    processed_h5_files_list = data_loader.get_files_in_directory(
        Path(config["paths"]["processed_files_path"]), ".feather")
    randomize_data(processed_h5_files_list,
                   Path(config["paths"]["training_files_path"]))

    all_progress_bar.update()
    all_progress_bar.set_postfix({"Status": "Completed"})


FEATURES_INFO = data_loader.load_json('config/features_info.json')

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
    # Option to delete old files that are generated in the in-between steps of preparation (.csv, .feather)
    ap.add_argument(
        "-override",
        required=False,
        action='store_true',
        help="Override old files, if left out just adds files not yet present")
    # Adjust verbosity level from 0 to 4 -v ... -vvvv
    ap.add_argument("-verbose",
                    "-v",
                    required=False,
                    action='count',
                    help="Verbosity intensity | Only 2 tiers")

    args = ap.parse_args()

    if (args.verbose == None):
        args.verbose = 0

    start_time_all = time.time()

    # Load config
    config_path = 'config/dataset_config.json'
    if (args.config):
        config_path = args.config

    config = data_loader.load_json(config_path)

    if (args.mode != None):
        if (args.mode == 'all'):
            print("Prepare entire dataset...")
            prepare_dataset()
        elif (args.mode == 'parse'):
            print("Only parsing...")
            demo_files_list = data_loader.get_files_in_directory(
                Path(config["paths"]["demo_files_path"]), '.dem')
            parse_data(demo_files_list,
                       Path(config["paths"]["parsed_files_path"]),
                       override=args.override)
        elif (args.mode == 'preprocess'):
            print("Only preprocessing...")
            parsed_csv_files_list = data_loader.get_files_in_directory(
                Path(config["paths"]["parsed_files_path"]), ".csv")
            parsed_csv_files_list = list(
                filter(lambda name: "death" not in name.name,
                       parsed_csv_files_list))
            preprocess_data(parsed_csv_files_list,
                            Path(config["paths"]["processed_files_path"]),
                            override=args.override)
        elif (args.mode == 'randomize'):
            processed_h5_files_list = data_loader.get_files_in_directory(
                Path(config["paths"]["processed_files_path"]), ".feather")
            randomize_data(processed_h5_files_list,
                           Path(config["paths"]["training_files_path"]))

    # Time keeping TODO: Time keeping is erroneous
    end_time_all = time.time()

    elapsed_time_all = end_time_all - start_time_all
    print("Preparation process in mode " + args.mode + " took >>" +
          str(elapsed_time_all) + " seconds<<")
