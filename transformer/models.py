import torch.nn as nn
import torch
import torch.nn.functional as F
import math, copy, re

device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda")

###########################  Stage 1 #####################################################
class Embedding(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super(Embedding, self).__init__()

        self.embed = nn.Embedding(vocab_size, embed_dim)

    def forward(self, x):
        out = self.embed(x)
        return out

class PositionalEmbedding(nn.Module):
    def __init__(self, max_seq_len, embed_model_dim):
        super(PositionalEmbedding, self).__init__()
        
        self.embed_dim = embed_model_dim
        pe = torch.zeros(max_seq_len, self.embed_dim)
        for pos in range(max_seq_len):
            for i in range(0, self.embed_dim, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i) / self.embed_dim)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1)) / self.embed_dim)))
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # Make embeddings relatively larger
        x = x * math.sqrt(self.embed_dim)
        # Add constant to embedding
        seq_len = x.size(1)
        x = x + torch.autograd.Variable(self.pe[:, :seq_len], requires_grad = False)
        return x
    
# As found in https://github.com/evelinehong/Transformer_Relative_Position_PyTorch/blob/master/relative_position.py
class RelativePosition(nn.Module):
    def __init__(self, num_units, max_relative_position):
        super().__init__()

        self.num_units = num_units
        self.max_relative_position = max_relative_position
        self.embeddings_table = nn.Parameter(torch.zeros(max_relative_position * 2 + 1, num_units, dtype = torch.float32, device = device))
        nn.init.xavier_uniform_(self.embeddings_table)

    def forward(self, length_q, length_k):
        range_vec_q = torch.arange(length_q)
        range_vec_k = torch.arange(length_k)
        distance_mat = range_vec_k[None, :] - range_vec_q[:, None]
        distance_mat_clipped = torch.clamp(distance_mat, -self.max_relative_position, self.max_relative_position)
        final_mat = distance_mat_clipped + self.max_relative_position
        final_mat = final_mat.clone().detach()
        embeddings = self.embeddings_table[final_mat]
        return embeddings

############################### Stage 2 - Multi-head attention ###############################################

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim = 512, n_heads = 8):
        super(MultiHeadAttention, self).__init__()

        self.embed_dim = embed_dim  # 512 dim
        self.n_heads = n_heads  # 8
        self.single_head_dim = int(self.embed_dim / self.n_heads)  # 512/8 = 64  . each key,query, value will be of 64d

        # key,query and value matrixes    #64 x 64
        self.query_matrix = nn.Linear(self.single_head_dim, self.single_head_dim, bias = False)  # single key matrix for all 8 keys #512x512
        self.key_matrix = nn.Linear(self.single_head_dim, self.single_head_dim, bias = False)
        self.value_matrix = nn.Linear(self.single_head_dim, self.single_head_dim, bias = False)
        self.out = nn.Linear(self.n_heads * self.single_head_dim, self.embed_dim)

    def forward(self, key, query, value, mask = None):  # batch_size x sequence_length x embedding_dim    # 32 x 10 x 512
        batch_size = key.size(0)
        seq_length = key.size(1)

        # query dimension can change in decoder during inference.
        # so we cant take general seq_length
        seq_length_query = query.size(1)

        # 32x10x512
        key = key.view(batch_size, seq_length, self.n_heads, self.single_head_dim)  # batch_size x sequence_length x n_heads x single_head_dim = (32x10x8x64)
        query = query.view(batch_size, seq_length_query, self.n_heads, self.single_head_dim)  # (32x10x8x64)
        value = value.view(batch_size, seq_length, self.n_heads, self.single_head_dim)  # (32x10x8x64)

        k = self.key_matrix(key)  # (32x10x8x64)
        q = self.query_matrix(query)
        v = self.value_matrix(value)

        q = q.transpose(1, 2)  # (batch_size, n_heads, seq_len, single_head_dim)    # (32 x 8 x 10 x 64)
        k = k.transpose(1, 2)  # (batch_size, n_heads, seq_len, single_head_dim)
        v = v.transpose(1, 2)  # (batch_size, n_heads, seq_len, single_head_dim)

        # computes attention
        # adjust key for matrix multiplication
        k_adjusted = k.transpose(-1, -2)  # (batch_size, n_heads, single_head_dim, seq_ken)  #(32 x 8 x 64 x 10)
        product = torch.matmul(q, k_adjusted)  # (32 x 8 x 10 x 64) x (32 x 8 x 64 x 10) = #(32x8x10x10)

        # fill those positions of product matrix as (-1e20) where mask positions are 0
        if mask is not None:
            mask = mask.to(device = product.device) 
            product = product.masked_fill(mask == 0, float("-1e20"))

        # divising by square root of key dimension
        product = product / math.sqrt(self.single_head_dim)  # / sqrt(64)

        # applying softmax
        scores = F.softmax(product, dim = -1)

        # mutiply with value matrix
        scores = torch.matmul(scores, v)  ##(32x8x 10x 10) x (32 x 8 x 10 x 64) = (32 x 8 x 10 x 64)

        # concatenated output
        concat = scores.transpose(1, 2).contiguous().view(batch_size, seq_length_query, self.single_head_dim * self.n_heads)  # (32x8x10x64) -> (32x10x8x64)  -> (32,10,512)

        output = self.out(concat)  # (32,10,512) -> (32,10,512)

        return output

# As found in https://github.com/evelinehong/Transformer_Relative_Position_PyTorch/blob/master/relative_position.py
class MultiHeadAttentionRelativePosition(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout):
        super().__init__()
        
        assert hid_dim % n_heads == 0
        
        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.head_dim = hid_dim // n_heads
        self.max_relative_position = 2

        self.relative_position_k = RelativePosition(self.head_dim, self.max_relative_position)
        self.relative_position_v = RelativePosition(self.head_dim, self.max_relative_position)

        self.fc_q = nn.Linear(hid_dim, hid_dim)
        self.fc_k = nn.Linear(hid_dim, hid_dim)
        self.fc_v = nn.Linear(hid_dim, hid_dim)
        
        self.fc_o = nn.Linear(hid_dim, hid_dim)
        
        self.dropout = nn.Dropout(dropout)
        
        self.scale = torch.sqrt(torch.tensor([self.head_dim], dtype = torch.float32, device = device))
        
    def forward(self, key, query, value, mask = None):
        #query = [batch size, query len, hid dim]
        #key = [batch size, key len, hid dim]
        #value = [batch size, value len, hid dim]
        batch_size = query.shape[0]
        len_k = key.shape[1]
        len_q = query.shape[1]
        len_v = value.shape[1]

        query = self.fc_q(query)
        key = self.fc_k(key)
        value = self.fc_v(value)

        r_q1 = query.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        r_k1 = key.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        attn1 = torch.matmul(r_q1, r_k1.permute(0, 1, 3, 2)) 

        r_q2 = query.permute(1, 0, 2).contiguous().view(len_q, batch_size*self.n_heads, self.head_dim)
        r_k2 = self.relative_position_k(len_q, len_k)
        attn2 = torch.matmul(r_q2, r_k2.transpose(1, 2)).transpose(0, 1)
        attn2 = attn2.contiguous().view(batch_size, self.n_heads, len_q, len_k)
        attn = (attn1 + attn2) / self.scale

        if mask is not None:
            mask = mask.to(device = attn.device) 
            attn = attn.masked_fill(mask == 0, float("-1e20"))

        attn = self.dropout(torch.softmax(attn, dim = -1))

        #attn = [batch size, n heads, query len, key len]
        r_v1 = value.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        weight1 = torch.matmul(attn, r_v1)
        r_v2 = self.relative_position_v(len_q, len_v)
        weight2 = attn.permute(2, 0, 1, 3).contiguous().view(len_q, batch_size*self.n_heads, len_k)
        weight2 = torch.matmul(weight2, r_v2)
        weight2 = weight2.transpose(0, 1).contiguous().view(batch_size, self.n_heads, len_q, self.head_dim)

        x = weight1 + weight2
        
        #x = [batch size, n heads, query len, head dim]
        
        x = x.permute(0, 2, 1, 3).contiguous()
        
        #x = [batch size, query len, n heads, head dim]
        
        x = x.view(batch_size, -1, self.hid_dim)
        
        #x = [batch size, query len, hid dim]
        
        x = self.fc_o(x)
        
        #x = [batch size, query len, hid dim]
        
        return x

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, expansion_factor = 4, n_heads = 8, activation = "ReLU", norm_first = False, relative_attention = False):
        super(TransformerBlock, self).__init__()
        
        if relative_attention == True:
            self.attention = MultiHeadAttentionRelativePosition(embed_dim, n_heads, 0)
        else:
            self.attention = MultiHeadAttention(embed_dim, n_heads)

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm_first = norm_first

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, expansion_factor * embed_dim),
            (nn.GELU() if activation.lower() == "gelu" else nn.ReLU()),
            nn.Linear(expansion_factor * embed_dim, embed_dim))

        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.2)

    def forward(self, key, query, value):

        if self.norm_first == True:
            attention_out = self.attention(key, query, self.norm1(value))  # 32x10x512
            attention_residual_out = attention_out + value  # 32x10x512
            norm1_out = self.dropout1(attention_residual_out)  # 32x10x512

            feed_fwd_out = self.feed_forward(self.norm2(norm1_out))  # 32x10x512 -> #32x10x2048 -> 32x10x512
            feed_fwd_residual_out = feed_fwd_out + norm1_out  # 32x10x512
            norm2_out = self.dropout2(feed_fwd_residual_out)  # 32x10x512

        else:
            attention_out = self.attention(key, query, value)  # 32x10x512
            attention_residual_out = attention_out + value  # 32x10x512
            norm1_out = self.dropout1(self.norm1(attention_residual_out))  # 32x10x512

            feed_fwd_out = self.feed_forward(norm1_out)  # 32x10x512 -> #32x10x2048 -> 32x10x512
            feed_fwd_residual_out = feed_fwd_out + norm1_out  # 32x10x512
            norm2_out = self.dropout2(self.norm2(feed_fwd_residual_out))  # 32x10x512

        return norm2_out

class TransformerEncoder(nn.Module):
    def __init__(self, seq_len, vocab_size, embed_dim, num_layers = 2, expansion_factor = 4, n_heads = 8, activation = "ReLU", norm_first = False, relative_attention = False):
        super(TransformerEncoder, self).__init__()

        self.embedding_layer = Embedding(vocab_size, embed_dim)
        self.positional_encoder = PositionalEmbedding(seq_len, embed_dim)

        self.layers = nn.ModuleList([TransformerBlock(embed_dim = embed_dim, expansion_factor = expansion_factor, n_heads = n_heads, activation = activation, norm_first = norm_first, relative_attention = relative_attention) for i in range(num_layers)])

    def forward(self, x):
        embed_out = self.embedding_layer(x)
        out = self.positional_encoder(embed_out)
        for layer in self.layers:
            out = layer(out, out, out)

        return out  # 32x10x512

class DecoderBlock(nn.Module):
    def __init__(self, embed_dim, expansion_factor = 4, n_heads = 8, activation = "ReLU", norm_first = False, relative_attention = False):
        super(DecoderBlock, self).__init__()

        if relative_attention == True:
            self.attention = MultiHeadAttentionRelativePosition(embed_dim, n_heads, 0)
        else:
            self.attention = MultiHeadAttention(embed_dim, n_heads)

        self.norm = nn.LayerNorm(embed_dim)
        if norm_first == True:
            self.norm2 = nn.LayerNorm(embed_dim)
            self.norm3 = nn.LayerNorm(embed_dim)
        self.norm_first = norm_first
        self.dropout = nn.Dropout(0.2)
        self.transformer_block = TransformerBlock(embed_dim = embed_dim, expansion_factor = expansion_factor, n_heads = n_heads, activation = activation, norm_first = norm_first, relative_attention = relative_attention)

    def forward(self, key, query, x, mask):
        if self.norm_first == True:
            # We need to pass mask only to first attention
            attention = self.attention(self.norm(x), self.norm2(x), self.norm3(x), mask = mask)  # 32x10x512
            value = self.dropout(attention + x)

        else:
            # We need to pass mask only to first attention
            attention = self.attention(x, x, x, mask = mask)  # 32x10x512
            value = self.dropout(self.norm(attention + x))

        out = self.transformer_block(key, query, value)

        return out

class TransformerDecoder(nn.Module):
    def __init__(self, target_vocab_size, embed_dim, seq_len, num_layers = 2, expansion_factor = 4, n_heads = 8, activation = "ReLU", norm_first = False, relative_attention = False):
        super(TransformerDecoder, self).__init__()

        self.word_embedding = nn.Embedding(target_vocab_size, embed_dim)
        self.position_embedding = PositionalEmbedding(seq_len, embed_dim)

        self.layers = nn.ModuleList([DecoderBlock(embed_dim, expansion_factor = expansion_factor, n_heads = n_heads, activation = activation, norm_first = norm_first, relative_attention = relative_attention) for _ in range(num_layers)])
        self.fc_out = nn.Linear(embed_dim, target_vocab_size)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x, enc_out, mask):

        x = self.word_embedding(x)  # 32x10x512
        x = self.position_embedding(x)  # 32x10x512
        x = self.dropout(x)

        for layer in self.layers:
            x = layer(enc_out, x, enc_out, mask)

        out = F.softmax(self.fc_out(x))

        return out

class Transformer(nn.Module):
    def __init__(self, embed_dim, src_vocab_size, target_vocab_size, seq_len, num_layers = 2, expansion_factor = 4, n_heads = 8, activation = "ReLU", norm_first = False, relative_attention = False):
        super(Transformer, self).__init__()

        self.target_vocab_size = target_vocab_size

        self.encoder = TransformerEncoder(seq_len = seq_len, vocab_size = src_vocab_size, embed_dim = embed_dim, num_layers = num_layers, expansion_factor = expansion_factor, n_heads = n_heads, activation = activation, norm_first = norm_first, relative_attention = relative_attention)
        self.decoder = TransformerDecoder(target_vocab_size = target_vocab_size, embed_dim = embed_dim, seq_len = seq_len, num_layers = num_layers, expansion_factor = expansion_factor, n_heads = n_heads, activation = activation, norm_first = norm_first, relative_attention = relative_attention)

    def make_trg_mask(self, trg):
        batch_size, trg_len = trg.shape
        # Returns the lower triangular part of matrix filled with ones
        trg_mask = torch.tril(torch.ones((trg_len, trg_len))).expand(batch_size, 1, trg_len, trg_len)
        return trg_mask

    def decode(self, src, trg):
        trg_mask = self.make_trg_mask(trg)
        enc_out = self.encoder(src)
        out_labels = []
        batch_size, seq_len = src.shape[0], src.shape[1]
        # outputs = torch.zeros(seq_len, batch_size, self.target_vocab_size)
        out = trg
        for i in range(seq_len):  # 10
            out = self.decoder(out, enc_out, trg_mask)  # bs x seq_len x vocab_dim
            # taking the last token
            out = out[:, -1, :]

            out = out.argmax(-1)
            out_labels.append(out.item())
            out = torch.unsqueeze(out, axis=0)

        return out_labels

    def forward(self, src, trg):
        trg_mask = self.make_trg_mask(trg)
        enc_out = self.encoder(src)

        outputs = self.decoder(trg, enc_out, trg_mask)
        return outputs