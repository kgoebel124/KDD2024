# Setup

* Set up conda environment with required packages in requirements.txt ```pip install -r requirements.txt```
* Install pytorch using ```conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch```
* Install GPytorch using ```pip install gpytorch```

# Training and running the model

Train the model:
* run from command line as "python main.py --{argument} {option}"
* e.g.: python main.py --name example_run --experiment concat_embedding --include_lte --include_pheno
* IMPORTANT NOTE: the dataset is currently not public, data must be added to ./data/valid/ for training to run properly

Arguments:
* name: can be any string, just determines where files are saved. If not included, will default to current date and time
* experiment: model type to use (embedding, multihead)
* pretrained_path: location of model file, excluding will mean training a new model
* epochs: number of training epochs
* lr: learning rate for training
* data_path: location of data, note that this folder is currently empty in the repository
* include_lte: flag that indicates training with LTE data
* include_pheno: flag that indicates training with phenology
* phenos: phenological events to include
* These are only the most relevant arguments, see the beginning of main.py for more arguments and the valid choices for each argument
