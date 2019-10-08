# General instructions

## Excution location

The root directory during execution should be the **csgo_encounter_prediction folder**.
Otherwise issues with paths and directories may arise.

## Configuration

Inside the config you can find multiple variables:

### Path configuration

- `demo_files_path`: Location of downloaded and unpacked .dem files of CS:GO matches
- `parsed_files_path`: Location of parsed .dem files stored as .csv
- `processed_files_path`: Location of .h5 files that have undergone preprocessing and whose data is ready for training
- `dataset_files_path`: Location of randomized chunks of data that are actually used for training

All paths are relative to the root directory. All paths may point to the same location, but for visibility purposes different paths are recommended.

## Execution

### Preprocessing

After setting up the configuration, simply execute the `preprocessing/create_dataset.py` for complete parsing, preprocessing and randomization of the .dem files.

For separate execution of the steps use the appropiately named files in the `preprocessing` folder:

- `parsing.js`: Run to parse all .dem files specified in config. Output are **.csv** files of matches
- `preprocess.py`: Add and remove all neccesary features of data for training. Output are **.h5** of matches
- `randomize.py`: Randomize match files and generate chunks as **.h5** files. These can be used for training.

WIP - work in progress