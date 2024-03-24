import torch
import torch.nn as nn
import torch.nn.functional as F
import random

from attention_models import BahdanauAttention

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

        # Initialise Embedding layer
        self.embedding = nn.Embedding(input_size, hidden_size)
        # Initialise GRU layer
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers = self.num_layers, batch_first = True, bidirectional = self.bidirectional)
        # Initialise Layer normalisation
        self.layer_normaliser = nn.LayerNorm(hidden_size)
        # Initialise Dropout
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, input, hidden = None):
        # Apply dropout to input embeddings
        embedded = self.dropout(self.embedding(input))
        # Forward pass through bidirectional GRU layer
        output, hidden = self.gru(embedded, hidden)

        # Sum bidirectional outputs
        if self.bidirectional == True:
            output = (output[:, :, :self.hidden_size] + output[:, :, self.hidden_size:])

        # Apply layer normalisation to output
        if self.layer_norm == True:
            output = self.layer_normaliser(output)

        return output, hidden

class DecoderRNN(nn.Module):
    SOS_token = 0
    EOS_token = 1

    def __init__(self, hidden_size, output_size, num_layers = 1, max_seq_len = 50):
        super(DecoderRNN, self).__init__()
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first = True)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, encoder_outputs, encoder_hidden, target_tensor = None):
        batch_size = encoder_outputs.size(0)
        decoder_input = torch.empty(batch_size, 1, dtype = torch.long, device = device).fill_(self.SOS_token)
        #decoder_hidden = encoder_hidden
        decoder_hidden = encoder_hidden[-self.num_layers:]
        decoder_outputs = []

        for i in range(self.max_seq_len):
            decoder_output, decoder_hidden = self.forward_step(decoder_input, decoder_hidden)
            decoder_outputs.append(decoder_output)

            if target_tensor is not None:
                # Teacher forcing: Feed the target as the next input
                decoder_input = target_tensor[:, i].unsqueeze(1) # Teacher forcing
            else:
                # Without teacher forcing: use its own predictions as the next input
                _, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze(-1).detach()  # detach from history as input

        decoder_outputs = torch.cat(decoder_outputs, dim = 1)
        decoder_outputs = F.log_softmax(decoder_outputs, dim = -1)
        return decoder_outputs, decoder_hidden, None # We return `None` for consistency in the training loop

    def forward_step(self, input, hidden):
        output = self.embedding(input)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.out(output)
        return output, hidden
    
class AttnDecoderGRU(nn.Module):
    SOS_token = 0
    EOS_token = 1

    def __init__(self, hidden_size, output_size, max_seq_len = 50, dropout_p = 0.1, attention_type = "bahdanau"):
        super(AttnDecoderGRU, self).__init__()
        self.max_seq_len = max_seq_len
        self.embedding = nn.Embedding(output_size, hidden_size)

        if attention_type == "bahdanau":
            self.attention = BahdanauAttention(hidden_size)
            self.attention_function = self.forward_step_bahdanau
            self.gru = nn.GRU(2 * hidden_size, hidden_size, batch_first = True)

        self.out = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, encoder_outputs, encoder_hidden, target_tensor = None):
        batch_size = encoder_outputs.size(0)
        decoder_input = torch.empty(batch_size, 1, dtype = torch.long, device = device).fill_(self.SOS_token)
        decoder_hidden = encoder_hidden
        decoder_outputs = []
        attentions = []

        for i in range(self.max_seq_len):

            decoder_output, decoder_hidden, attn_weights = self.attention_function(
                decoder_input, decoder_hidden, encoder_outputs
            )
            decoder_outputs.append(decoder_output)
            attentions.append(attn_weights)

            if target_tensor is not None:
                # Teacher forcing: Feed the target as the next input
                decoder_input = target_tensor[:, i].unsqueeze(1) # Teacher forcing
            else:
                # Without teacher forcing: Use its own predictions as the next input
                _, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze(-1).detach()  # Detach from history as input

        decoder_outputs = torch.cat(decoder_outputs, dim = 1)
        decoder_outputs = F.log_softmax(decoder_outputs, dim = -1)
        if attentions[0] is not None:
            attentions = torch.cat(attentions, dim = 1)
        else:
            attentions = None

        return decoder_outputs, decoder_hidden, attentions

    def forward_step_bahdanau(self, input, hidden, encoder_outputs):
        embedded =  self.dropout(self.embedding(input))

        query = hidden.permute(1, 0, 2)
        context, attn_weights = self.attention(query, encoder_outputs)
        input_gru = torch.cat((embedded, context), dim=2)

        output, hidden = self.gru(input_gru, hidden)
        output = self.out(output)

        return output, hidden, attn_weights

######################################
### Improved Encoders and Decoders ###
######################################

# Use Multilayer Bidirectional GRU and Layer Normalisation
class EncoderRNN_improved(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers = 2, dropout_p = 0.1):
        super(EncoderRNN_improved, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Initialise Embedding layer
        self.embedding = nn.Embedding(input_size, hidden_size)
        # Initialise (Bidirectional) GRU layer
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers = self.num_layers, batch_first = True, bidirectional = True)
        # Initialise Layer normalisation
        self.layer_norm = nn.LayerNorm(hidden_size)
        # Initialise Dropout
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, input):
        # Apply dropout to input embeddings
        embedded = self.dropout(self.embedding(input))
        # Forward pass through bidirectional GRU layer
        output, hidden = self.gru(embedded)
        # Sum bidirectional outputs
        output = output[:, :, :self.hidden_size] + output[:, :, self.hidden_size:]
        # Apply layer normalisation to output
        output = self.layer_norm(output)

        return output, hidden

class DecoderRNN_improved(nn.Module):
    SOS_token = 0
    EOS_token = 1

    def __init__(self, hidden_size, output_size, max_seq_len = 50, scheduled_sampling_k = 0.95):
        super(DecoderRNN_improved, self).__init__()
        self.max_seq_len = max_seq_len
        self.scheduled_sampling_k = scheduled_sampling_k
        self.scheduled_sampling_prob = 1.0  # Initial probability
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first = True)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, encoder_outputs, encoder_hidden, target_tensor = None):
        batch_size = encoder_outputs.size(0)
        decoder_input = torch.empty(batch_size, 1, dtype = torch.long, device=device).fill_(self.SOS_token)
        decoder_hidden = encoder_hidden
        decoder_outputs = []

        for i in range(self.max_seq_len):
            decoder_output, decoder_hidden = self.forward_step(decoder_input, decoder_hidden)
            decoder_outputs.append(decoder_output)

            if target_tensor is not None and random.random() < self.scheduled_sampling_prob:
                # Use teacher forcing
                decoder_input = target_tensor[:, i].unsqueeze(1)  # Teacher forcing
            else:
                # Use its own predictions
                _, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze(-1).detach()  # detach from history as input
            
            # Update scheduled sampling probability using exponential decay
            scheduled_sampling_prob *= self.scheduled_sampling_k

        decoder_outputs = torch.cat(decoder_outputs, dim = 1)
        decoder_outputs = F.log_softmax(decoder_outputs, dim = -1)
        return decoder_outputs, decoder_hidden, None  # We return `None` for consistency in the training loop

    def forward_step(self, input, hidden):
        output = self.embedding(input)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.out(output)
        return output, hidden
    
class AttnDecoderGRU_improved(nn.Module):
    SOS_token = 0
    EOS_token = 1

    def __init__(self, hidden_size, output_size, max_seq_len = 50, dropout_p = 0.1, attention_type = "bahdanau"):
        super(AttnDecoderGRU_improved, self).__init__()
        self.max_seq_len = max_seq_len
        self.embedding = nn.Embedding(output_size, hidden_size)
        if attention_type == "bahdanau":
            self.attention = BahdanauAttention(hidden_size)
            self.attention_function = self.forward_step_bahdanau
            self.gru = nn.GRU(2 * hidden_size, hidden_size, batch_first = True)

        self.out = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, encoder_outputs, encoder_hidden, target_tensor = None):
        batch_size = encoder_outputs.size(0)
        decoder_input = torch.empty(batch_size, 1, dtype = torch.long, device = device).fill_(self.SOS_token)
        decoder_hidden = encoder_hidden
        decoder_outputs = []
        attentions = []

        for i in range(self.max_seq_len):

            decoder_output, decoder_hidden, attn_weights = self.attention_function(
                decoder_input, decoder_hidden, encoder_outputs
            )
            decoder_outputs.append(decoder_output)
            attentions.append(attn_weights)

            if target_tensor is not None:
                # Teacher forcing: Feed the target as the next input
                decoder_input = target_tensor[:, i].unsqueeze(1) # Teacher forcing
            else:
                # Without teacher forcing: use its own predictions as the next input
                _, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze(-1).detach()  # detach from history as input

        decoder_outputs = torch.cat(decoder_outputs, dim = 1)
        decoder_outputs = F.log_softmax(decoder_outputs, dim = -1)
        if attentions[0] is not None:
            attentions = torch.cat(attentions, dim = 1)
        else:
            attentions = None

        return decoder_outputs, decoder_hidden, attentions

    def forward_step_bahdanau(self, input, hidden, encoder_outputs):
        embedded =  self.dropout(self.embedding(input))

        query = hidden.permute(1, 0, 2)
        context, attn_weights = self.attention(query, encoder_outputs)
        input_gru = torch.cat((embedded, context), dim = 2)

        output, hidden = self.gru(input_gru, hidden)
        output = self.out(output)

        return output, hidden, attn_weights