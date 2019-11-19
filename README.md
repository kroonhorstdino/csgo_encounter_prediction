# General instructions

**_WARNING_**: All features described in this document are still very much WIP and probably don't work as described if at all,
!

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
- `processed_files_path`: Location of .feather files that have undergone preprocessing and whose data is ready for training
- `dataset_files_path`: Location of randomized chunks of data that are actually used for training

All paths are relative to the root directory. All paths may point to the same location, but for visibility purposes different paths are recommended.

After this setup the dataset preparation can begin

## How to run dataset preparation

### Do it in one go...

After setting up the configuration, simply execute the `preparation/prepare_dataset.py` for complete parsing, preprocessing and randomization of the .dem files.

By using the `-config PATH` option, the script will use the specified config. Otherwise it will use the default config `config/prep_config.json`

Verbosity of logging is specified by the option `-v` in stages up to `-vvvv`

### ... or one step at a time

For separate execution of the steps use the argument option **`-mode`** of `prepration/prepare_dataset.py`:

- `parse`: Run to parse all .dem files specified in config. Output is **.csv** files of matches
- `preprocess`: Add and remove all neccesary features of data for training. Output is **.feather** files of matches
- `randomize`: Randomize all matches and generate chunks as **.feather** files. These can be used for training.

Example:

`py create_dataset.py -mode preprocess -vv -config ./config/special.json`

With `py create_dataset.py --help` a help page can be displayed.

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