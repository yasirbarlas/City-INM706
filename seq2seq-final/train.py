import os
import time
import torch
import wandb
import torch.nn as nn
from torch import optim
import matplotlib.pyplot as plt

import numpy as np
import random

from dataset import TranslationDataset
from logger import Logger
from models import EncoderRNN, DecoderRNN
from attention_models import SelfAttention

from torch.utils.data import DataLoader, random_split

from utils import parse_arguments, read_settings, calculate_bleu, calculate_nist, scheduled_sampling

device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda")

print(device)

def train_epoch(train_dataloader, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, output_lang, teacher_forcing_ratio = 0.5, grad_clip = None, self_attention = None):
    encoder.train()
    decoder.train()
    
    total_loss = 0
    total_bleu_score = 0
    total_nist_score = 0
    
    for data in train_dataloader:
        input_sentence, input_tensor, target_tensor = data

        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        encoder_outputs, encoder_hidden = encoder(input_tensor)
        
        if self_attention is not None:
            encoder_outputs, att = self_attention(encoder_outputs)
        
        decoder_outputs, _, _ = decoder(encoder_outputs, encoder_hidden, target_tensor, teacher_forcing_ratio = teacher_forcing_ratio)

        loss = criterion(decoder_outputs.view(-1, decoder_outputs.size(-1)), target_tensor.view(-1))
        
        loss.backward()

        if isinstance(grad_clip, int) or isinstance(grad_clip, float):
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), grad_clip)
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), grad_clip)

        encoder_optimizer.step()
        decoder_optimizer.step()

        total_loss += loss.item()

        # Calculate BLEU score
        bleu_score = calculate_bleu(decoder_outputs, target_tensor)
        total_bleu_score += bleu_score

        # Calculate NIST score
        nist_score = calculate_nist(decoder_outputs, target_tensor)
        total_nist_score += nist_score

    if self_attention is not None:
        plot_attention_self(att[0, :, :])

    return total_loss / len(train_dataloader), total_bleu_score / len(train_dataloader), total_nist_score / len(train_dataloader)

def validate_epoch(val_dataloader, encoder, decoder, criterion):
    encoder.eval()
    decoder.eval()

    total_loss = 0
    total_bleu_score = 0
    total_nist_score = 0
    
    with torch.no_grad():
        for data in val_dataloader:
            input_sentence, input_tensor, target_tensor = data

            encoder_outputs, encoder_hidden = encoder(input_tensor)
            decoder_outputs, _, _ = decoder(encoder_outputs, encoder_hidden, None) # Do not input the target tensor as this is validation

            loss = criterion(decoder_outputs.view(-1, decoder_outputs.size(-1)), target_tensor.view(-1))

            total_loss += loss.item()

            # Calculate BLEU score
            bleu_score = calculate_bleu(decoder_outputs, target_tensor)
            total_bleu_score += bleu_score

            # Calculate NIST score
            nist_score = calculate_nist(decoder_outputs, target_tensor)
            total_nist_score += nist_score

    return total_loss / len(val_dataloader), total_bleu_score / len(val_dataloader), total_nist_score / len(val_dataloader)

def train(train_dataloader, val_dataloader, model, encoder, decoder, n_epochs, logger, input_lang, output_lang, learning_rate = 0.001, optimizer = "adam", criterion = "negative-log", teacher_forcing_ratio = "none", linear_tf_decay = False, grad_clip = None, self_attention = False):
    #val_losses = [float("inf")]
    val_bleus = [0]
    counter = 0

    # Adam
    if optimizer.lower() == "adam":
        encoder_optimizer = optim.Adam(encoder.parameters(), lr = learning_rate)
        decoder_optimizer = optim.Adam(decoder.parameters(), lr = learning_rate)
    # RAdam
    elif optimizer.lower() == "radam":
        encoder_optimizer = optim.RAdam(encoder.parameters(), lr = learning_rate)
        decoder_optimizer = optim.RAdam(decoder.parameters(), lr = learning_rate)
    # SGD Nesterov
    elif optimizer.lower() == "sgd":
        encoder_optimizer = optim.SGD(encoder.parameters(), lr = learning_rate, nesterov = True, momentum = 0.9)
        decoder_optimizer = optim.SGD(decoder.parameters(), lr = learning_rate, nesterov = True, momentum = 0.9)
    
    starting_epoch = 1

    if model is not None:
        encoder_optimizer.load_state_dict(model["encoder_optimizer_state_dict"])
        decoder_optimizer.load_state_dict(model["decoder_optimizer_state_dict"])
        starting_epoch = int(model["epoch"])

    # Loss function
    if criterion.lower() == "negative-log":
        criterion = nn.NLLLoss()
    elif criterion.lower() == "cross-entropy":
        criterion = nn.CrossEntropyLoss()

    # Self-Attention
    if self_attention == True:
        self_attention = SelfAttention(128, 128).to(device)

    # Training loop
    for epoch in range(starting_epoch, n_epochs + 1):
        # Teacher forcing decay/constant
        if (isinstance(teacher_forcing_ratio, int) or isinstance(teacher_forcing_ratio, float)):
            if linear_tf_decay == True:
                teacher_forcing_ratio = scheduled_sampling(epoch, n_epochs, start_ratio = teacher_forcing_ratio)
        # Using probabilities so if teacher forcing is not enabled, any number above 1 will disable teacher forcing
        else:
            teacher_forcing_ratio = 2
        
        # Train and Validation Loss and BLEU score
        train_loss, train_bleu, train_nist = train_epoch(train_dataloader, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, output_lang, teacher_forcing_ratio = teacher_forcing_ratio, grad_clip = grad_clip, self_attention = self_attention)
        val_loss, val_bleu, val_nist = validate_epoch(val_dataloader, encoder, decoder, criterion)

        val_bleus.append(val_bleu)

        print(f"Epoch: {epoch} / {n_epochs}, Train Loss {train_loss}, Validation Loss {val_loss}, Train BLEU {train_bleu}, Validation BLEU {val_bleu}, Train NIST {train_nist}, Validation NIST {val_nist}")

        logger.log({"train_loss": train_loss})
        logger.log({"validation_loss": val_loss})
        logger.log({"train_bleu": train_bleu})
        logger.log({"validation_bleu": val_bleu})
        logger.log({"train_nist": train_nist})
        logger.log({"validation_nist": val_nist})
        logger.log({"teacher_forcing_ratio": teacher_forcing_ratio})

        if (val_bleu + 0.0001) > val_bleus[-2]:
            # Restart patience (improvement in validation loss)
            counter = 0

            # Create checkpoint folder
            if not os.path.exists("checkpoints"):
                os.makedirs("checkpoints")

            # Save the model checkpoint
            checkpoint_name = f"checkpoint_seq2seq_latest.pth"
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
        
        elif (val_bleu + 0.0001) < val_bleus[-2]:
            # Add one to patience
            counter += 1

            if val_bleu > val_bleus[-2]:
                # Create checkpoint folder
                if not os.path.exists("checkpoints"):
                    os.makedirs("checkpoints")

                # Save the model checkpoint
                checkpoint_name = f"checkpoint_seq2seq_latest.pth"
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

def plot_attention_self(attentions):
    timestamp = int(time.time())
    filename = f"attention_{timestamp}.pdf"

    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(attentions.clone().detach().cpu().numpy(), cmap = "bone")
    fig.colorbar(cax)

    # Save the figure to a wandb artifact
    wandb.log({"attention_matrix": wandb.Image(fig)})

    # Save the figure to computer
    plt.savefig(filename, bbox_inches = "tight")

    # Close the figure to prevent it from being displayed in the notebook
    plt.close(fig)

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
    model_settings = settings.get("Model", {})
    train_settings = settings.get("Train", {})
    logger_settings = settings.get("Logger", {})
    checkpoint_settings = settings.get("Checkpoint", {})
    print(model_settings)
    print(train_settings)
    print(logger_settings)
    print(checkpoint_settings)

    # Initialise 'wandb' for logging
    wandb_logger = Logger(logger_settings["logger_name"], project = logger_settings["project_name"])
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
    encoder = EncoderRNN(dataset.input_lang.n_words, model_settings["hidden_dim"], bidirectional = model_settings["encoder_bidirect"], num_layers = model_settings["num_layers_encoder"], layer_norm = model_settings["layer_norm"]).to(device)
    decoder = DecoderRNN(model_settings["hidden_dim"], dataset.output_lang.n_words, num_layers = model_settings["num_layers_decoder"], use_lstm = model_settings["use_lstm_decoder"], max_seq_len = model_settings["max_seq_length"]).to(device)

    model = None

    # Use checkpoint weights
    if checkpoint_settings["checkpoint"].lower() != "none":
        model = torch.load(checkpoint_settings["checkpoint"], map_location = device)
        encoder.load_state_dict(model["encoder_state_dict"])
        decoder.load_state_dict(model["decoder_state_dict"])

    self_attention = None

    # Enable self-attention
    if model_settings["attention"].lower() == "self":
        self_attention = True

    # Training loop
    train(train_dataloader, val_dataloader, model, encoder, decoder, train_settings["epochs"], logger, dataset.input_lang, dataset.output_lang, learning_rate = train_settings["learning_rate"], optimizer = train_settings["optimizer"], criterion = train_settings["loss_function"], teacher_forcing_ratio = train_settings["teacher_forcing_ratio"], linear_tf_decay = train_settings["linear_tf_decay"], grad_clip = train_settings["grad_clip"], self_attention = self_attention)

if __name__ == "__main__":
    main()