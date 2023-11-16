import torch.nn as nn
from torch.nn.functional import softmax

import math

# Scaled Dot-Product Attention
def scaled_dot_product_attention(query, key, value):
    # Q @ KT
    attention = query @ key.T
    print(attention)

    # Scaling
    attention = attention / math.sqrt(key.dim())
    print(attention)

    # SoftMax
    attention = softmax(attention)
    print(attention)

    # @ V
    attention = attention @ value
    return attention


# MultiHeadAttention
class MultiHeadAttention(nn.Module):
    def __init__(self):
        pass

    def forward(self):
        pass


# MaskedMultiHeadAttention
## Decoder용 MultiHeadAttention
### 입력된 index에 따라 그 이후의 모든 단어는 inf로 처리
class MaskedMultiHeadAttention(nn.Module):
    def __init__(self):
        pass

    def forward(self):
        pass



# Encoder
class Encoder(nn.Module):
    def __init__(self):
        pass

    def forward(self):
        pass


# Decoder
class Decoder(nn.Module):
    def __init__(self):
        pass

    def forward(self):
        pass


# Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self):
        pass

    def forward(self):
        pass


# Embedding