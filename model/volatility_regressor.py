import torch
import torch.nn as nn
from model.transcript.components.conference_encoder import ConferenceEncoder

class VolatilityRegressor(nn.Module):
    def __init__(self, utt_emb_dim, seq_emb_dim):
        super(VolatilityRegressor, self).__init__()
        
        self.role_emb_size = 3
        self.role_embedder = nn.Embedding(3, self.role_emb_size)
        
        self.conference_encoder = ConferenceEncoder()
        self.linear_proj = nn.Linear(self.role_emb_size + utt_emb_dim + 1, self.conference_encoder.hidden_size)
        
        self.fc1 = nn.Linear(seq_emb_dim + self.conference_encoder.hidden_size, 1024)
        self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(1024)
        
        self.fc2 = nn.Linear(1024, 512)
        self.bn2 = nn.BatchNorm1d(512)
        
        self.fc3 = nn.Linear(512, 256)
        self.bn3 = nn.BatchNorm1d(256)
        
        self.fc4 = nn.Linear(256, 1)
        
        
    def forward(self, x):
        (seq_emb, utt_emb, role_tokens, part_tokens) = x

        role_emb = self.role_embedder(role_tokens)
        part_tokens = part_tokens.unsqueeze(-1)
        
        utterance_emb = torch.cat((utt_emb, role_emb, part_tokens), dim=-1)
        utterance_emb = self.linear_proj(utterance_emb)
        conference_emb = self.conference_encoder(utterance_emb)
        
        x = torch.cat((seq_emb, conference_emb), dim=-1)
        
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.bn1(x)
        
        x = self.fc2(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.bn2(x)
        
        x = self.fc3(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.bn3(x)
        
        x = self.fc4(x)
        
        return x