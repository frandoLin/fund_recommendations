# models/sequential_attention_model.py
import torch
import torch.nn as nn

class SequentialAttentionDNN(nn.Module):
    def __init__(self, region_vocab_size, strategy_vocab_size, embed_dim=8):
        super().__init__()
        self.region_embed = nn.Embedding(region_vocab_size, embed_dim)
        self.strategy_embed = nn.Embedding(strategy_vocab_size, embed_dim)

        input_dim = 4 + 2 * embed_dim  # numerical (4) + embeddings (2*8)
        self.attention_weights = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x_num, region_idx, strategy_idx):
        region_emb = self.region_embed(region_idx)
        strategy_emb = self.strategy_embed(strategy_idx)
        x = torch.cat([x_num, region_emb, strategy_emb], dim=1)

        # Optional: apply attention weights (optional in this version)
        # weights = torch.softmax(self.attention_weights(x), dim=0)
        # x = x * weights

        return self.encoder(x)


