# INM706: Deep Learning for Sequence Analysis

Note: This repository and its contents support the coursework of the INM706 module at City, University of London.

## Background

We use sequence-to-sequence (Seq2Seq) and transformer models for machine translation on an English to French dataset.

## Datasets

We use the 'Europarl v7' dataset for training/validation, which was the standard for training models in WMT 2015. The dataset is extracted from proceedings of the European Parliament, with translations in many languages. The dataset can be found [here](https://www.statmt.org/europarl/). Our test datasets can be found [here](https://www.statmt.org/wmt09/translation-task.html).

We have also uploaded the datasets we used ourselves, which may be easier to download. These can be downloaded [here](https://cityuni-my.sharepoint.com/:f:/g/personal/yasir-zubayr_barlas_city_ac_uk/EhvejfheMSdKhGURMvvLf-oBhpNXOFp2tkVRx0sx_hIfdQ?e=Dz5c4b). The exact datasets used are in "fr-en.zip".

### Requirements
- Python 3.9+
- [PyTorch (with CUDA for GPU usage)](https://pytorch.org/get-started/locally/)
- All other requirements listed in [**requirements.txt**](requirements.txt)

### Training
The training process involves optimising the Seq2Seq and Transformer models to generate accurate translations of English sentences. You can customise various hyperparameters such as batch size, learning rate, optimiser, and loss function in the `config.yaml` file. Please refer to the `config.yaml` file for a full list of hyperparameters.

### Checkpoints
Our checkpoints can be downloaded [here](https://cityuni-my.sharepoint.com/:f:/g/personal/yasir-zubayr_barlas_city_ac_uk/EuqShgL4qCNLq6IhTXPTM7QBM-_HcyvVtQNR-jPBJavyYA?e=HnwuQF). Each folder contains a 'readme.txt' file containing the hyperparameters used to train the models. You will need these if you edit the code later, otherwise we have already filled them in appropriately for you. You only need to replace the dummy checkpoints in this repository with those found [here](https://cityuni-my.sharepoint.com/:f:/g/personal/yasir-zubayr_barlas_city_ac_uk/EuqShgL4qCNLq6IhTXPTM7QBM-_HcyvVtQNR-jPBJavyYA?e=HnwuQF).

### File Structure
- `train.py`: File to initiate the training loop for the respective model, which uses the hyperparameters in the `config.yaml` file.
- `config.yaml`: File to edit hyperparameters. Additional hyperparameters are included for choosing project and logger names for 'wandb'.
- `models.py`: Contains the definitions of the seq2seq/transformer models.
- `dataset.py`: Creates our datasets in the relevant PyTorch format.
- `utils.py`: Utility functions for calculating BLEU, NIST, etc.
- `logger.py`: Logger class for logging training metrics to 'wandb'.
- `checkpoints/`: Directory to save model checkpoints during training, and can be used for inference.
- `inference.ipynb`: Jupyter Notebook used for inference. Allows for a sentence/test dataset to be input and to examine the results.

### Usage
1. We recommend first downloading/cloning the whole repository, though if you wish to work only with the baseline model you do not need the `/seq2seq-final` folder, and vice versa for working with our final model and with transformers. We also recommend sticking to the folder structure found in this repository, otherwise you will need to make a few edits indicating where the datasets and checkpoints can be found.

2. Secondly, you should ensure that all the libraries listed in the [**requirements.txt**](requirements.txt) file have been installed on your environment (you may wish to create a new environment from scratch to avoid dependency conflicts). You can use your favourite method to install these libraries, such as through using the `pip install -r requirements.txt` command in your terminal. You must also install [PyTorch](https://pytorch.org/get-started/locally/), ideally the CUDA version if you wish to work with a GPU. Follow the instructions on the [PyTorch website](https://pytorch.org/get-started/locally/) for the installation.

3. Now, you will need to download a training and validation set (these could also be a single dataset as done in our work). In our work, we used 'Europarl-v7' as our training and validation set, at 80% and 20% respectively. We have created a folder for the dataset already in our repository, and you can download the datasets [here](https://cityuni-my.sharepoint.com/:f:/g/personal/yasir-zubayr_barlas_city_ac_uk/EhvejfheMSdKhGURMvvLf-oBhpNXOFp2tkVRx0sx_hIfdQ?e=Dz5c4b). This folder also includes our test datasets, which are only used in the respective `inference.ipynb`. We recommend deleting our dataset folder and replacing them with the folder that you download. In other words, download the dataset(s), delete our dummy dataset folder, and extract the dataset contents (which should be in a folder with the same name as our dummy folder) and put them in the same location as where the dummy dataset folders were (see repository, you essentially need the dataset folders outside of the actual code folders).

4. After completing all of the previous steps, you can safely run the `train.py` file from your chosen folder. You may edit the `config.yaml` file with your chosen hyperparameters to use during training, including adding any checkpoint paths (such as those that we provide).

6. **Optional**: Should you wish to use our checkpoints, you need to download them [here](https://cityuni-my.sharepoint.com/:f:/g/personal/yasir-zubayr_barlas_city_ac_uk/EuqShgL4qCNLq6IhTXPTM7QBM-_HcyvVtQNR-jPBJavyYA?e=HnwuQF) for your choice of model. You need to edit the 'config.yaml' file with the checkpoint path if you interested in using our checkpoints for the relevant model. These checkpoints should be placed in the `/checkpoints` folder for the relevant model folder.

7. **IMPORTANT NOTE**: The code assumes that you open your workspace in either the `/seq2seq-baseline`, `/seq2seq-final`, `/transformer-baseline`, or `/transformer-final` model folders in your integrated development environment (IDE), rather than opening the whole folder containing all of the datasets, etc. You can edit these to your preferences, but we recommend following the folder structure set out in this repository for ease.

### Inference
We provide a Jupyter Notebook (`inference.ipynb`) for inference. You must have our checkpoints downloaded and added to the `checkpoints/` folder, though you can edit the paths as you wish to the checkpoints inside the code. You can alternatively use your own checkpoints.

### Weights & Biases ('wandb')
If you have not used 'wandb' previously, you will be prompted to enter your API key into the terminal. You need a (free) 'wandb' account if not already made, and you can find further instructions [here](https://docs.wandb.ai/quickstart).
