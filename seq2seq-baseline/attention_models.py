import torch
import math
import torch.nn as nn
import torch.nn.functional as F
    
class SelfAttention(nn.Module):
    def __init__(self, input_size, out_size):
        super(SelfAttention, self).__init__()
        self.dk_size = out_size
        self.query_linear = nn.Linear(in_features = input_size, out_features = out_size)
        self.key_linear = nn.Linear(in_features = input_size, out_features = out_size)
        self.value_linear = nn.Linear(in_features = input_size, out_features = out_size)
        self.softmax = nn.Softmax(dim = -1)

    def forward(self, input_vector):
        query_out = F.relu(self.query_linear(input_vector))
        key_out = F.relu(self.key_linear(input_vector))

        value_out = F.relu(self.value_linear(input_vector))
        att = torch.bmm(query_out, key_out.transpose(1, 2))

        out_q_k = torch.div(att, math.sqrt(self.dk_size))
        softmax_q_k = self.softmax(out_q_k)
        out_combine = torch.bmm(softmax_q_k, value_out)
        return out_combine, att