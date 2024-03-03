import time
import os
import torch
import wandb
import torch.nn as nn
from torch import optim
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import numpy as np
import random

from dataset import TranslationDataset
from logger import Logger
from models import EncoderRNN, DecoderRNN

from torch.utils.data import Dataset, DataLoader, random_split

from utils import parse_arguments, read_settings

device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda")


def evaluate(encoder, decoder, input_tensor, output_lang):
    EOS_token = 1
    with torch.no_grad():
        encoder_outputs, encoder_hidden = encoder(input_tensor)
        decoder_outputs, decoder_hidden, decoder_attn = decoder(encoder_outputs, encoder_hidden)

        _, topi = decoder_outputs.topk(1)
        decoded_ids = topi.squeeze()

        decoded_words = []
        for idx in decoded_ids:
            if idx.item() == EOS_token:
                decoded_words.append("<EOS>")
                break
            decoded_words.append(output_lang.index2word[idx.item()])
    return decoded_words, decoder_attn

def train_epoch(dataloader, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, output_lang):
    total_loss = 0
    for data in dataloader:
        input_sentence, input_tensor, target_tensor = data

        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        encoder_outputs, encoder_hidden = encoder(input_tensor)
        decoder_outputs, _, _ = decoder(encoder_outputs, encoder_hidden, target_tensor)

        loss = criterion(
            decoder_outputs.view(-1, decoder_outputs.size(-1)),
            target_tensor.view(-1)
        )
        loss.backward()

        encoder_optimizer.step()
        decoder_optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def train(train_dataloader, encoder, decoder, n_epochs, logger, output_lang, learning_rate = 0.001, print_every = 100, plot_every = 100):
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.Adam(encoder.parameters(), lr = learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr = learning_rate)
    criterion = nn.NLLLoss()

    for epoch in range(1, n_epochs + 1):
        loss = train_epoch(train_dataloader, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion,output_lang)
        print_loss_total += loss
        plot_loss_total += loss

        if epoch % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print(f"Epoch: {epoch} / {n_epochs}, Loss {print_loss_avg}")

        if epoch % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            logger.log({"loss_avg": plot_loss_avg})
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

        if (epoch % 5 == 0) or (epoch == n_epochs):
            # Create checkpoint folder
            if not os.path.exists("checkpoints"):
                os.makedirs("checkpoints")

            # Save the model checkpoint
            checkpoint_name = f"checkpoint_epoch_{epoch}.pt"
            checkpoint_path = os.path.join("checkpoints", checkpoint_name)
            torch.save({
                'encoder_state_dict': encoder.state_dict(),
                'decoder_state_dict': decoder.state_dict(),
                'encoder_optimizer_state_dict': encoder_optimizer.state_dict(),
                'decoder_optimizer_state_dict': decoder_optimizer.state_dict(),
                'epoch': epoch,
                'loss': loss
            }, checkpoint_path)


def main():
    # Set random seed for reproducibility
    randomer = 50
    torch.manual_seed(randomer)
    random.seed(randomer)
    np.random.seed(randomer)

    generator = torch.Generator().manual_seed(randomer)

    args = parse_arguments()

    # Read settings from the YAML file
    settings = read_settings(args.config)

    # Access and use the settings as needed
    model_settings = settings.get('model', {})
    train_settings = settings.get('train', {})
    print(model_settings)
    print(train_settings)

    # Initialise 'wandb' for logging
    wandb_logger = Logger(f"inm706_translation_seq2seq_v1", project = "inm706_cwkk")
    logger = wandb_logger.get_logger()
    
    # Get dataset
    dataset = TranslationDataset(lang1 = "en", lang2 = "fr", max_seq_len = model_settings["max_seq_length"], reverse = True)

    # Define the sizes for training and validation sets
    train_size = int(0.75 * len(dataset))
    val_size = len(dataset) - train_size

     # Use 'random_split' to split the dataset into training and validation sets
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator = generator)

    # Create data loaders for each set
    train_dataloader = DataLoader(train_dataset, batch_size = train_settings["batch_size"])
    val_dataloader = DataLoader(val_dataset, batch_size = train_settings["batch_size"])

    # Define encoder and decoder
    encoder = EncoderRNN(dataset.input_lang.n_words, model_settings["hidden_dim"]).to(device)
    decoder = DecoderRNN(model_settings["hidden_dim"], dataset.output_lang.n_words, max_seq_len = model_settings["max_seq_length"]).to(device)

    # Run training loop
    train(train_dataloader, encoder, decoder, train_settings["epochs"], logger, dataset.output_lang, print_every = 1, plot_every = 1)

    return

if __name__ == "__main__":
    main()
