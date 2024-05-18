import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda")

# Default Encoder
class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, bidirectional = False, num_layers = 1, layer_norm = False, dropout_p = 0.1):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.layer_norm = layer_norm

        self.num_directions = 1
        if self.bidirectional == True:
            self.num_directions = 2

        # Initialise Embedding layer
        self.embedding = nn.Embedding(input_size, hidden_size)
        # Initialise GRU layer
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers = self.num_layers, batch_first = True, bidirectional = self.bidirectional)
        # Initialise Layer normalisation
        if self.layer_norm == True:
            self.layer_normaliser = nn.LayerNorm(hidden_size)
        # Initialise Dropout
        self.dropout = nn.Dropout(dropout_p)

    # From https://github.com/chrisvdweth/nus-cs4248x/blob/master/3-neural-nlp/src/rnn.py
    def _concat_directions(self, s, batch_size):
        # s.shape = (num_layers*num_directions, batch_size, hidden_size)
        X = s.view(self.num_layers, self.num_directions, batch_size, self.hidden_size)
        # X.shape = (num_layers, num_directions, batch_size, hidden_size)
        X = X.permute(0, 2, 1, 3)
        # X.shape = (num_layers, batch_size, num_directions, hidden_size)
        return X.contiguous().view(self.num_layers, batch_size, -1) 

    def forward(self, input, hidden = None):
        batch_size, _ = input.shape

        # Apply dropout to input embeddings
        embedded = self.dropout(self.embedding(input))
        # Forward pass through GRU layer
        output, hidden = self.gru(embedded, hidden)

        # Sum bidirectional outputs and concatenate hidden
        if self.bidirectional == True:
            output = (output[:, :, :self.hidden_size] + output[:, :, self.hidden_size:])
            #hidden = self._concat_directions(hidden, batch_size)

        # Apply layer normalisation to output
        if self.layer_norm == True:
            output = self.layer_normaliser(output)

        return output, hidden

class DecoderRNN(nn.Module):
    SOS_token = 0
    EOS_token = 1

    def __init__(self, hidden_size, output_size, num_layers = 1, use_lstm = False, max_seq_len = 50):
        super(DecoderRNN, self).__init__()
        self.num_layers = num_layers
        self.use_lstm = use_lstm
        self.max_seq_len = max_seq_len
        self.embedding = nn.Embedding(output_size, hidden_size)

        if self.use_lstm == True:
            self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers = self.num_layers, batch_first = True)

        if self.use_lstm == False:
            self.gru = nn.GRU(hidden_size, hidden_size, num_layers = self.num_layers, batch_first = True)

        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, encoder_outputs, encoder_hidden, target_tensor = None, teacher_forcing_ratio = 0.9):
        batch_size = encoder_outputs.size(0)
        decoder_input = torch.empty(batch_size, 1, dtype = torch.long, device = device).fill_(self.SOS_token)

        if self.use_lstm == True:
            decoder_hidden = (encoder_hidden[-self.num_layers:], torch.zeros_like(encoder_hidden[-self.num_layers:]))
        else:
            decoder_hidden = encoder_hidden[-self.num_layers:]

        decoder_outputs = []

        for i in range(self.max_seq_len):
            decoder_output, decoder_hidden = self.forward_step(decoder_input, decoder_hidden)

            decoder_outputs.append(decoder_output)

            # Teacher forcing: Feed the target as the next input
            if (target_tensor is not None) and (torch.rand(1).item() < teacher_forcing_ratio):
                decoder_input = target_tensor[:, i].unsqueeze(1) # Teacher forcing
            # Without teacher forcing: Use its own predictions as the next input
            else:
                _, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze(-1).detach() # Detach from history as input

        decoder_outputs = torch.cat(decoder_outputs, dim = 1)
        decoder_outputs = F.log_softmax(decoder_outputs, dim = -1)

        return decoder_outputs, decoder_hidden, None # We return 'None' for consistency in the training loop, or the attentions

    def forward_step(self, input, hidden):
        output = self.embedding(input)
        output = F.relu(output)

        if self.use_lstm == True:
            output, hidden = self.lstm(output, hidden)

        if self.use_lstm == False:
            output, hidden = self.gru(output, hidden)

        output = self.out(output)
        return output, hidden