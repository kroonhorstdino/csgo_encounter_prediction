# General instructions

**_WARNING_**: All features described in this document are still very much WIP!

## Excution location

The root directory for executing programs should always be **`/csgo_encounter_prediction/`**.
Otherwise issues with paths and directories may arise.

# Collect data

## Demo files

For the dataset only .dem files of CS:GO matches are accepted.

For downloading CS:GO pro matches you can use my own little program that downloads the last x public pro matches: [Source Demo Downloader](https://github.com/kroonhorstdino/source_demo_replaydownloader)

#  Dataset preparation __(Parsing, Preprocessing, Randomization)__

## Configuration

Inside the `prep_config.json` file you can find multiple variables related to preparing the dataset:

- `demo_files_path`: Location of downloaded and unpacked .dem files of CS:GO matches
- `parsed_files_path`: Location of parsed .dem files stored as .csv
- `processed_files_path`: Location of .h5 files that have undergone preprocessing and whose data is ready for training
- `dataset_files_path`: Location of randomized chunks of data that are actually used for training

All paths are relative to the root directory. All paths may point to the same location, but for visibility purposes different paths are recommended.

After this setup the dataset preparation can begin

## How to run dataset preparation

### Do it in one go...

After setting up the configuration, simply execute the `preprocessing/prepare_dataset.py` for complete parsing, preprocessing and randomization of the .dem files.

### ... or one step at a time

For separate execution of the steps use the appropiately named files in the `preprocessing` folder:

- `parsing.js`: Run to parse all .dem files specified in config. Output is **.csv** files of matches
- `preprocess.py`: Add and remove all neccesary features of data for training. Output is **.h5** files of matches
- `randomize.py`: Randomize all matches and generate chunks as **.h5** files. These can be used for training.

For each program, new paths can be specified so that the default config paths are ignored

For example: `python preprocessing/parsing.js ../other_dir/other_demo_file_dir/ ./other_parsed_files`
This input will parse **.dem** files from another folder and put it into another target directory. Relative paths can be used.

## Hyperparameter search

WIP

Traingin configs can be either automatically or manually configured.

# Training the network

### Configuration

As with the dataset preparation `train_config.json` is the configuration for training the network.

- `shared_layers`: Array that holds sizes of layers in the shared network
- `dense_layers`: Array for sizes of dense network 

- `learning_rate`: Learning rate of network

- `batch_size`: Amount of samples in one minibatch

- `epoch_size`: Size of epochs


## How to run it

Network training is done by executing the **`train.py`** file in the root folder. It is also possible to specify a special config for training. If no path is given the program will use the default `config/train_config.json`.

For example:

    python train.py ./config/special_config.json

After some epochs models of the network are saved to `./models/`

# Testing and using the network

WIP