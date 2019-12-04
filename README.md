# Instructions for use

## Excution location

The root directory for executing programs should always be **`/csgo_encounter_prediction/`**
Otherwise issues with paths and directories may arise.

## Demo files

The datataset has to consist of recorded CS:GO matches in form of `.dem` files.

#  Dataset preparation __(Parsing, Preprocessing, Randomization)__

## Configuration

Inside the `dataset_config.json` file you can find multiple variables related to preparing the dataset:

- `demo_files_path`: Location of downloaded and unpacked .dem files of CS:GO matches
- `parsed_files_path`: Location of parsed .dem files stored as .csv
- `processed_files_path`: Location of .feather files that have undergone preprocessing and whose data is ready for training
- `training_files_path`: Location of randomized chunks of data that are actually used for training
- `death_time_window`: Time window for classification labels
- `chunk_row_size`: Row size of randomized chunks.

All paths are relative to the root directory. All paths may point to the same location, but for visibility purposes different paths are recommended.
After this setup the dataset preparation can begin.

## How to run dataset preparation

### Do it in one go...

After setting up the configuration,execute the `preparation/prepare_dataset.py` with the `-mode all` argument for complete parsing, preprocessing and randomization of the .dem files.

By using the `-config PATH` option, the script will use the specified config. Otherwise it will use the default config `config/dataset_config.json`

Verbosity of logging is specified by the option `-v` in stages up to `-vvvv`

If the `-override` option is left out, no files are replaced. Othweise all files are being replaced during the process.

### ... or one step at a time

For separate execution of the steps use the argument option **`-mode`** of `prepration/prepare_dataset.py`:

- `parse`: Run to parse all .dem files specified in config. Output is **.csv** files of matches
- `preprocess`: Add and remove all neccesary features of data for training. Output is **.feather** files of matches
- `randomize`: Randomize all matches and generate chunks as **.feather** files. These can be used for training.

Example:

`python preparation/prepare_dataset.py -mode preprocess  -config ./config/special.json -vv -override`

With `py prepare_dataset.py -h` a help page is be displayed.

## Hyperparameter search

Hyperparameter search is perfomed by running the script `training/hyperparameter_search.py`.
Runs a certain amount of named training runs. Generates configs for each run and saves them to `config/<NAME_OF_RUN>`. The results are also saved to tensorboard
The parameters of the hyperparameter search have to be adjusted inside the script.

# Training the network

### Configuration

As with the dataset preparation `train_config.json` is the configuration for training the network.

- `shared_layers`, `dense_layers`: Arrays that contain sizes of shared and dense layers
- `feature_set`, `label_set`: Feature and label sets for training (seen in `features_info.json`
- `lr`: Learning rate of network
- `batch_size`: Amount of samples in one batch
- `num_epoch`: Amount of epochs
- `checkpoint_epoch`: Interval at which to save state of model

## How to run it

Network training is done by executing the **`training/csgotest.py`** script By deault the `config/train_config.json` config is used. In addition a name can be specified for the naming run.

For example:

    python train.py -trainconf config/train_config.json -name test

Trained models are saved to `./models/<NAME_OF_RUN>`

For a description of the different parameters enter:
    
    python train.py -h

## View training results

All training runs are recorded and can be viewed in real time in tensorboard.

If installed type:

    tensorboard --logdir=results/
    
to start tensorboard.
