# INM706: Deep Learning for Sequence Analysis

Note: This repository and its contents support the coursework of the INM706 module at City, University of London.

## Background

## Datasets

We use the 'Europarl v7' dataset for training/validation, which was the standard for training models in WMT 2015. The dataset is extracted from proceedings of the European Parliament, with translations in many languages. The dataset can be found [here](https://www.statmt.org/europarl/). Our test datasets can be found [here](https://www.statmt.org/wmt09/translation-task.html).

We have also uploaded the datasets we used ourselves, which may be easier to download. These can be downloaded [here](https://cityuni-my.sharepoint.com/:f:/g/personal/yasir-zubayr_barlas_city_ac_uk/EhvejfheMSdKhGURMvvLf-oBhpNXOFp2tkVRx0sx_hIfdQ?e=Dz5c4b). The exact datasets used are in "fr-en.zip".

### Requirements
- Python 3.9+
- [PyTorch (with CUDA for GPU usage)](https://pytorch.org/get-started/locally/)
- All other requirements listed in [**requirements.txt**]()

### Checkpoints
Our checkpoints can be downloaded [here](). Each folder contains a 'readme.txt' file containing the hyperparameters used to train the models. You will need these if you edit the code later, otherwise we have already filled them in appropriately for you. You only need to replace the dummy checkpoints in this repository with those found [here]().

### File Structure
- `train.py`: File to initiate the training loop for the respective model, which uses the hyperparameters in the `config.yaml` file.
- `config.yaml`: File to edit hyperparameters. Additional hyperparameters are included for choosing project and logger names for 'wandb'.
- `models.py`: Contains the definitions of the seq2seq/transformer models.
- `dataset.py`: Creates our datasets in the relevant PyTorch format.
- `utils.py`: Utility functions for calculating BLEU, NIST, etc.
- `logger.py`: Logger class for logging training metrics to 'wandb'.
- `checkpoints/`: Directory to save model checkpoints during training, and can be used for inference.
- `inference.ipynb`: Jupyter Notebook used for inference. Allows for a sentence/test dataset to be input and to examine the results.

### Inference
We provide a Jupyter Notebook (`inference.ipynb`) for inference. You must have our checkpoints downloaded and added to the `checkpoints/` folder, though you can edit the paths as you wish to the checkpoints inside the code. You can alternatively use your own checkpoints.

### Weights & Biases ('wandb')
If you have not used 'wandb' previously, you will be prompted to enter your API key into the terminal. You need a (free) 'wandb' account if not already made, and you can find further instructions [here](https://docs.wandb.ai/quickstart).
