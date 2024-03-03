import time
import torch
import wandb
import torch.nn as nn
from torch import optim
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from dataset import TranslationDataset
from logger import Logger
from models import EncoderRNN, DecoderRNN

from torch.utils.data import Dataset, DataLoader

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


def plot_attention(input_sentence, output_words, attentions):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(attentions.cpu().numpy(), cmap="bone")
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels([""] + input_sentence.split(" ") +
                       ["<EOS>"], rotation=90)
    ax.set_yticklabels([""] + output_words)

    # Show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    # Save the figure to a wandb artifact
    wandb.log({"attention_matrix": wandb.Image(fig)})

    # Close the figure to prevent it from being displayed in the notebook
    plt.close(fig)

def plot_attention_self(attentions):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(attentions.clone().detach().cpu().numpy(), cmap="bone")
    fig.colorbar(cax)

    # Save the figure to a wandb artifact
    wandb.log({"attention_matrix": wandb.Image(fig)})

    # Close the figure to prevent it from being displayed in the notebook
    plt.close(fig)


def plot_and_show_attention(encoder, decoder, input_sentence, input_tensor, output_lang_voc):
    output_words, attentions = evaluate(encoder, decoder, input_tensor, output_lang_voc)
    plot_attention(input_sentence, output_words, attentions[0, :len(output_words), :])


def train_epoch(dataloader, encoder, decoder, encoder_optimizer,
          decoder_optimizer, criterion, output_lang, self_att=None, plot_attention=False):

    total_loss = 0
    for data in dataloader:
        input_sentence, input_tensor, target_tensor = data

        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        encoder_outputs, encoder_hidden = encoder(input_tensor)
        if self_att is not None:
            encoder_outputs, att = self_att(encoder_outputs)
        decoder_outputs, _, _ = decoder(encoder_outputs, encoder_hidden, target_tensor)

        loss = criterion(
            decoder_outputs.view(-1, decoder_outputs.size(-1)),
            target_tensor.view(-1)
        )
        loss.backward()

        encoder_optimizer.step()
        decoder_optimizer.step()

        total_loss += loss.item()
    if plot_attention:
        plot_and_show_attention(encoder, decoder, input_sentence[0], input_tensor[0,:].unsqueeze(0), output_lang)
    if self_att is not None:
        plot_attention_self(att[0,:,:])


    return total_loss / len(dataloader)


def train(
        train_dataloader, encoder, decoder, n_epochs, logger, output_lang, learning_rate = 0.001,
        print_every = 100, plot_every = 100, plot_attention = False):
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.Adam(encoder.parameters(), lr = learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr = learning_rate)
    criterion = nn.NLLLoss()

    for epoch in range(1, n_epochs + 1):
        loss = train_epoch(train_dataloader, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion,output_lang, plot_attention=plot_attention)
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


def main():
    wandb_logger = Logger(f"inm706_translation_seq2seq_v1", project = "inm706_cwkk")
    logger = wandb_logger.get_logger()
    dataset = TranslationDataset("en", "fr", True)
    hidden_size = 128
    batch_size = 64
    n_epochs = 50

    train_dataloader = DataLoader(dataset, batch_size = batch_size)

    encoder = EncoderRNN(dataset.input_lang.n_words, hidden_size).to(device)
    decoder = DecoderRNN(hidden_size, dataset.output_lang.n_words).to(device)

    train(train_dataloader, encoder, decoder, n_epochs, logger, dataset.output_lang, print_every = 5, plot_every = 5, plot_attention = False)

    return

if __name__ == "__main__":
    main()
