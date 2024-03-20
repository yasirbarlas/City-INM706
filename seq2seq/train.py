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
from models import EncoderRNN, AttnDecoderGRU, DecoderRNN

from torch.utils.data import DataLoader, random_split

from utils import parse_arguments, read_settings, calculate_bleu, calculate_nist

device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda")

print(device)

def train_epoch(train_dataloader, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, output_lang, plot_attention = False):
    total_loss = 0
    total_bleu_score = 0
    total_nist_score = 0
    
    for data in train_dataloader:
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

        # Calculate BLEU score
        bleu_score = calculate_bleu(decoder_outputs, target_tensor)
        total_bleu_score += bleu_score

        # Calculate NIST score
        nist_score = calculate_nist(decoder_outputs, target_tensor)
        total_nist_score += nist_score

    if plot_attention:
        plot_and_show_attention(encoder, decoder, input_sentence[0], input_tensor[0, :].unsqueeze(0), output_lang)

    return total_loss / len(train_dataloader), total_bleu_score / len(train_dataloader), total_nist_score / len(train_dataloader)

def validate(val_dataloader, encoder, decoder, criterion, output_lang):
    total_loss = 0
    total_bleu_score = 0
    total_nist_score = 0
    
    for data in val_dataloader:
        input_sentence, input_tensor, target_tensor = data

        encoder_outputs, encoder_hidden = encoder(input_tensor)
        decoder_outputs, _, _ = decoder(encoder_outputs, encoder_hidden, target_tensor)

        loss = criterion(
            decoder_outputs.view(-1, decoder_outputs.size(-1)),
            target_tensor.view(-1)
        )

        total_loss += loss.item()

        # Calculate BLEU score
        bleu_score = calculate_bleu(decoder_outputs, target_tensor)
        total_bleu_score += bleu_score

        # Calculate NIST score
        nist_score = calculate_nist(decoder_outputs, target_tensor)
        total_nist_score += nist_score

    return total_loss / len(val_dataloader), total_bleu_score / len(val_dataloader), total_nist_score / len(val_dataloader)

def train(train_dataloader, val_dataloader, encoder, decoder, n_epochs, logger, input_lang, output_lang, learning_rate = 0.001, optimizer = "adam", criterion = "negative-log", plot_attention = False):
    val_losses = [float("inf")]
    counter = 0

    if optimizer == "adam":
        encoder_optimizer = optim.Adam(encoder.parameters(), lr = learning_rate)
        decoder_optimizer = optim.Adam(decoder.parameters(), lr = learning_rate)
    elif optimizer == "radam":
        encoder_optimizer = optim.RAdam(encoder.parameters(), lr = learning_rate)
        decoder_optimizer = optim.RAdam(decoder.parameters(), lr = learning_rate)
    
    if criterion == "negative-log":
        criterion = nn.NLLLoss()
    elif criterion == "cross-entropy":
        criterion = nn.CrossEntropyLoss()

    for epoch in range(1, n_epochs + 1):
        # Train and Validation Loss and BLEU score
        train_loss, train_bleu, train_nist = train_epoch(train_dataloader, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, output_lang, plot_attention = plot_attention)
        val_loss, val_bleu, val_nist = validate(val_dataloader, encoder, decoder, criterion, output_lang)

        val_losses.append(val_loss)

        print(f"Epoch: {epoch} / {n_epochs}, Train Loss {train_loss}, Validation Loss {val_loss}, Train BLEU {train_bleu}, Validation BLEU {val_bleu}, Train NIST {train_nist}, Validation NIST {val_nist}")

        logger.log({"train_loss": train_loss})
        logger.log({"validation_loss": val_loss})
        logger.log({"train_bleu": train_bleu})
        logger.log({"validation_bleu": val_bleu})
        logger.log({"train_nist": train_nist})
        logger.log({"validation_nist": val_nist})

        if (val_loss + 0.01) < val_losses[epoch - 1]:
            # Restart patience (improvement in validation loss)
            counter = 0

            # Create checkpoint folder
            if not os.path.exists("checkpoints"):
                os.makedirs("checkpoints")

            # Save the model checkpoint
            checkpoint_name = f"checkpoint_latest.pth"
            checkpoint_path = os.path.join("checkpoints", checkpoint_name)
            torch.save({
                "encoder_state_dict": encoder.state_dict(),
                "decoder_state_dict": decoder.state_dict(),
                "encoder_optimizer_state_dict": encoder_optimizer.state_dict(),
                "decoder_optimizer_state_dict": decoder_optimizer.state_dict(),
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_bleu": train_bleu,
                "val_bleu": val_bleu,
                "train_nist": train_nist,
                "val_nist": val_nist,
                "input_lang": input_lang,
                "output_lang": output_lang
            }, checkpoint_path)
        
        elif (val_loss + 0.01) > val_losses[epoch - 1]:
            # Add one to patience
            counter += 1
            # Patience reached, stop training (no significant improvement in validation loss after 5 epochs)
            if counter >= 5:
                break
    return

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

def plot_attention(input_sentence, output_words, attentions):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(attentions.cpu().numpy(), cmap = "bone")
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels([""] + input_sentence.split(" ") + ["<EOS>"], rotation = 90)
    ax.set_yticklabels([""] + output_words)

    # Show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    # Save the figure to a wandb artifact
    wandb.log({"attention_matrix": wandb.Image(fig)})

    # Close the figure to prevent it from being displayed in the notebook
    plt.close(fig)

def plot_and_show_attention(encoder, decoder, input_sentence, input_tensor, output_lang_voc):
    output_words, attentions = evaluate(encoder, decoder, input_tensor, output_lang_voc)
    plot_attention(input_sentence, output_words, attentions[0, :len(output_words), :])

def main():
    # Set random seed for reproducibility
    randomer = 50
    torch.manual_seed(randomer)
    torch.cuda.manual_seed_all(randomer)
    random.seed(randomer)
    np.random.seed(randomer)

    generator = torch.Generator().manual_seed(randomer)

    # Read settings from the YAML file
    args = parse_arguments()
    settings = read_settings(args.config)

    # Access and use the settings as needed
    model_settings = settings.get("model", {})
    train_settings = settings.get("train", {})
    print(model_settings)
    print(train_settings)

    # Initialise "wandb" for logging
    wandb_logger = Logger(f"inm706_translation_seq2seq_v1", project = "inm706_cwk")
    logger = wandb_logger.get_logger()
    
    # Get dataset
    dataset = TranslationDataset(lang1 = "en", lang2 = "fr", max_seq_len = model_settings["max_seq_length"], reverse = False)

    # Define the sizes for training and validation sets
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    # Split the dataset into training and validation sets
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator = generator)

    # Create data loaders for each set
    train_dataloader = DataLoader(train_dataset, batch_size = train_settings["batch_size"])
    val_dataloader = DataLoader(val_dataset, batch_size = train_settings["batch_size"])

    # Define encoder and decoder
    encoder = EncoderRNN(dataset.input_lang.n_words, model_settings["hidden_dim"]).to(device)
    
    # Get decoder (with or without attention) and train model
    if model_settings["attention"] != "none":
        decoder = AttnDecoderGRU(model_settings["hidden_dim"], dataset.output_lang.n_words, max_seq_len = model_settings["max_seq_length"], attention_type = model_settings["attention"]).to(device)
        train(train_dataloader, val_dataloader, encoder, decoder, train_settings["epochs"], logger, dataset.input_lang, dataset.output_lang, learning_rate = train_settings["learning_rate"], optimizer = train_settings["optimizer"], criterion = train_settings["loss_function"], plot_attention = True)

    else:
        decoder = DecoderRNN(model_settings["hidden_dim"], dataset.output_lang.n_words, max_seq_len = model_settings["max_seq_length"]).to(device)
        train(train_dataloader, val_dataloader, encoder, decoder, train_settings["epochs"], logger, dataset.input_lang, dataset.output_lang, learning_rate = train_settings["learning_rate"], optimizer = train_settings["optimizer"], criterion = train_settings["loss_function"], plot_attention = False)

    return

if __name__ == "__main__":
    main()