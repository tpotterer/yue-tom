import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torch
from model.positional_encoder import PositionalEncoder

class ConferenceEncoder(nn.Module):
    def __init__(self, transformer_hidden_size=128, nhead=4, num_layers=4, max_length=256, dropout=0.1):
        super(ConferenceEncoder, self).__init__()
        
        self.pos_enc = PositionalEncoder(transformer_hidden_size, max_len=max_length)
        self.encoder_layer = TransformerEncoderLayer(d_model=transformer_hidden_size, nhead=nhead, dropout=dropout)
        self.document_transformer = TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        
        self.hidden_size = transformer_hidden_size
        self.max_length = max_length
        
    def forward(self, x):
        # truncate to max tokens
        x = x[:self.max_length,:]
        x = self.document_transformer(x).mean(dim=0)
        
        return x