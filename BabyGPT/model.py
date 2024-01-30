import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


class LayerNorm(nn.Module):
    # D = dimensions
    # B = bias
    def __init__(self, D, B):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(D))
        self.bias = nn.Parameter(torch.ones(B))

    def forward(self, x):
        return F.layer_norm(
            input=x,
            normalized_shape=self.weight.shape,
            weight=self.weight,
            bias=self.bias,
            eps=1e-05
        )


class Attention(nn.Module):
    def __init__(self, config):
        super.__init__()
        # Terminology from paper: https://arxiv.org/pdf/1706.03762.pdf
        # d_model = config.embedding_dimension
        # h = config.num_heads
        # d_k = d_v = d_model / h

        assert config.embedding_dimension % config.num_heads == 0, \
            f"Embedding size ({config.embedding_dimension} should be a multiple of the number of attention heads ({config.num_heads}).)"

        # Ideally there are h linear projections of the matrices of size (d_model) x (d_k)
        # But, we can view it each component as a matrix.
        self.W_k = nn.Linear(config.embedding_dimension, config.embedding_dimension)
        self.W_q = nn.Linear(config.embedding_dimension, config.embedding_dimension)
        self.W_v = nn.Linear(config.embedding_dimension, config.embedding_dimension)
        self.attention_dropout = nn.Dropout(config.dropout_rate)
        
        # W_o dimensions: (h * d_v) x (d_model)
        self.W_o = nn.Linear(config.embedding_dimension, config.embedding_dimensions)
        self.multihead_dropout = nn.Dropout(config.dropout_rate)

        self.num_head = config.num_heads

        # Mask buffer parameter that ensures auto-regressive property of not attending tokens to the right
        # See https://arxiv.org/pdf/1706.03762.pdf, Sec 3.2.3
        self.register_buffer(
            "mask",
            # Tril consider lower half of the matrix
            torch.tril(torch.ones(config.block_size, config.block_size).view(1, 1, config.block_size, config.block_size))
        )

    def forward(self, x):
        # Notations
        # B = batch_size
        # S = sequence_length
        # E = embedding_dimension
        # H = num_heads
        batch_size, sequence_length, embedding_dimension = x.size()

        def get_projected_vector(input, layer):
            # (B, S, E) -> (B, S, E)
            output = layer(input)
            # (B, S, E) -> (B, S, H, E/H)
            output = output.view(batch_size, sequence_length, self.num_heads, embedding_dimension // self.num_heads)
            # (B, S, H, E/H) -> (B, H, S, E/H)
            output = torch.einsum("bshe -> bhse", output)
            return output
        
        k = get_projected_vector(x, self.W_k)
        q = get_projected_vector(x, self.W_q)
        v = get_projected_vector(x, self.W_v)

        k = torch.einsum("bhse -> bshe")
        attention = torch.einsum("bhse, bshe -> bhhe", q, k) * (1.0 / math.sqrt(k.size(-2)))
        attention = attention.masked_fill(
            # When value in the mask matrix is zero, set value in attention to -inf
            self.mask[:,:,sequence_length, sequence_length] == 0,
            float('-inf')
        )
        attention = F.softmax(attention)
        attention = self.attention_dropout(attention)
        attention = torch.einsum("bhhe, bhse -> bhse", attention, v)
        attention = torch.einsum("bhse->bshe", attention)
        # View throws if the tensor in not contiguous in memory after all the attention mechanism operations
        # See https://stackoverflow.com/questions/48915810/what-does-contiguous-do-in-pytorch
        attention = attention.contiguous().view(batch_size, sequence_length, embedding_dimension)
        
        y = self.W_o(attention)
        y = self.multihead_dropout(y)

        return y



class FancyFeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.linear_tranform_1 = nn.Linear(config.embedding_dimension, 4 * config.embedding_dimension, bias=config.bias)
        # Original transformer paper user relu
        # See https://arxiv.org/pdf/1706.03762.pdf, Sec 3.3
        # self.relu = nn.relu()
        # GPT-2 uses GELU
        self.gelu = nn.gelu()
        self.linear_transform_2 = nn.Linear(4 * config.embedding_dimension, config.embedding_dimension, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout_rate)

    
    def forward(self, x):
        x = self.linear_tranform_1(x)
        x = self.gelu(x)
        x = self.linear_transform_2(x)
        x = self.dropout(x)
        return x



class MyGPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.word_token_embedding = nn.Embedding(config.vocab_size, config.embedding_dimension)
        self.word_position_embedding = nn.Embedding(config.block_size, config.embedding_dimension)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.layer_norm_1 = LayerNorm(config.embedding_dimension, config.bias)
        self.attention = Attention(config)
        self.layer_norm_2 = LayerNorm(config.embedding_dimension, config.bias)
        self.feed_forward = FancyFeedForward(config)
        self.layer_norm_3 = LayerNorm(config.embedding_dimension, config.bias)
        self.linear_layer = nn.Linear(config.embedding_dimension, config.vocab_size, bias=False)

        # https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.apply
        self.apply(self._initialize_weights)


    def get_num_params(self):
        num_params = sum([torch.numel(p) for p in self.parameters()])
        return num_params

    def _initialize_weights(self, module):
        if isinstance(module, nn.Linear):
            # https://pytorch.org/docs/stable/nn.init.html#torch.nn.init.normal_
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input, teacher_forcing_target=None):
        device = input.device
        batch_size, token_embedding_size = input.size()

        # Calculate Embeddings
        positions = torch.arange(
            start=0, end=token_embedding_size, dtype=torch.long, device=device
        )
        token_embedding = self.word_token_embedding(input)
        position_embedding = self.word_position_embedding(positions)
        x = self.dropout(token_embedding + position_embedding)
        x = x + self.attention(self.layer_norm_1(x))
        x = x + self.feed_forward(self.layer_norm_2(x))
        x = self.layer_norm_3(x)

        if teacher_forcing_target:
            logits = self.linear_layer(x)
            # Batch size x vocab_size
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                target=teacher_forcing_target.view(-1),
                ignore_index=-1
            )
        else:
            logits = self.linear_layer(x[:, [-1], :])
            loss = None
        
        return logits, loss


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304
    num_layers: int = 4
    num_heads: int = 12
    embedding_dimension: int = 768
    dropout: float = 0.0
    bias: bool = False


