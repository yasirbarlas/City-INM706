import torch
import yaml
import os
import pandas as pd
import argparse

from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score
from nltk.translate.nist_score import sentence_nist
from nltk.translate.bleu_score import SmoothingFunction

device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda")


def save_checkpoint(epoch, model, model_name, optimizer):
    ckpt = {'epoch': epoch, 'model_weights': model.state_dict(), 'optimizer_state': optimizer.state_dict()}
    torch.save(ckpt, f"{model_name}_ckpt_{str(epoch)}.pth")


def load_checkpoint(model, file_name):
    ckpt = torch.load(file_name, map_location=device)
    model_weights = ckpt['model_weights']
    model.load_state_dict(model_weights) 
    print("Model's pretrained weights loaded!")

def parse_arguments():
    parser = argparse.ArgumentParser(description='Process settings from a YAML file.')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to YAML configuration file')
    return parser.parse_args()


def read_settings(config_path):
    folder_path = os.path.join(os.getcwd(), 'Transformer')
    config_path = os.path.join(folder_path, config_path)
    with open(config_path, 'r') as file:
        settings = yaml.safe_load(file)
    return settings

def calculate_bleu(output, target_tensor):
    bleu_scores = []
    for i in range(len(output)):
        predicted_indices = output[i].argmax(dim = -1).tolist()
        target_indices = target_tensor[i].tolist()

        # Remove padding and SOS token from target tensor
        if 1 in target_indices:
            target_indices = target_indices[:target_indices.index(1)]

        # Calculate BLEU score for each sequence
        bleu_score = sentence_bleu([target_indices], predicted_indices, smoothing_function = SmoothingFunction().method1)
        bleu_scores.append(bleu_score)

    # Average BLEU scores across all sequences
    avg_bleu_score = sum(bleu_scores) / len(bleu_scores)
    return avg_bleu_score

def calculate_nist(output, target_tensor, n = 4):
    nist_scores = []
    for i in range(len(output)):
        predicted_indices = output[i].argmax(dim = -1).tolist()
        target_indices = target_tensor[i].tolist()

        # Remove padding and SOS token from target tensor
        if 1 in target_indices:
            target_indices = target_indices[:target_indices.index(1)]

        # Calculate NIST score for each sequence
        nist_score = sentence_nist([target_indices], predicted_indices, n)
        nist_scores.append(nist_score)

    # Average NIST scores across all sequences
    avg_nist_score = sum(nist_scores) / len(nist_scores)
    return avg_nist_score
