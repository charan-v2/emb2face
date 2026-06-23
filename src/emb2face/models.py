import torch
import torch.nn as nn


class LinearAdapter(nn.Module):
    def __init__(self, dim: int = 512):
        super().__init__()
        self.fc = nn.Linear(dim, dim)

    def forward(self, x):
        return self.fc(x)


class MLPAdapter(nn.Module):
    def __init__(self, dim: int = 512, hidden_dim: int = 1024, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x):
        return self.net(x)


class ResidualMLPAdapter(nn.Module):
    def __init__(self, dim: int = 512, hidden_dim: int = 1024, dropout: float = 0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x):
        return x + self.block(x)


class PairDataset(torch.utils.data.Dataset):
    def __init__(self, df_subset, ada_embeddings, arc_embeddings):
        self.df = df_subset.reset_index(drop=True)
        self.ada = ada_embeddings
        self.arc = arc_embeddings

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        ada = self.ada[int(row["ada_index"])]
        arc = self.arc[int(row["arc_index"])]
        return torch.from_numpy(ada).float(), torch.from_numpy(arc).float()
