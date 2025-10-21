import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt
import math

# ======= 位置编码 =======
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


# ======= Transformer 模型 =======
class TimeSeriesTransformer(nn.Module):
    def __init__(
        self,
        input_features,
        dec_features,
        d_model=128,
        nhead=8,
        num_layers=2,
        dim_feedforward=256,
        dropout=0.1,
        max_len=500,
        output_dim=None,
    ):
        super().__init__()
        self.d_model = d_model
        self.encoder_input_layer = nn.Linear(input_features, d_model)
        self.decoder_input_layer = nn.Linear(dec_features, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len)

        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.output_layer = nn.Linear(d_model, output_dim or dec_features)
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        src = self.encoder_input_layer(src) * math.sqrt(self.d_model)
        src = self.positional_encoding(src)
        tgt = self.decoder_input_layer(tgt) * math.sqrt(self.d_model)
        tgt = self.positional_encoding(tgt)
        output = self.transformer(src, tgt, src_mask=src_mask, tgt_mask=tgt_mask)
        output = self.output_layer(output)
        return output


# ======= 数据生成：多特征正弦波 =======
def generate_multi_sine_dataset(seq_len=48, pred_len=6, num_samples=1000, num_features=5, noise_std=0.05):
    X, Y = [], []
    total_len = seq_len + pred_len
    for _ in range(num_samples):
        start = np.random.rand() * 2 * np.pi
        time = np.linspace(start, start + total_len * 0.1, total_len)
        data = []
        for f in range(num_features):
            freq = 1 + 0.2 * f
            data.append(np.sin(freq * time + np.random.rand()) + noise_std * np.random.randn(total_len))
        data = np.stack(data, axis=-1)  # shape: (total_len, num_features)
        X.append(data[:seq_len])
        Y.append(data[seq_len:])
    return np.array(X), np.array(Y)


class MultiFeatureDataset(Dataset):
    def __init__(self, seq_len=48, pred_len=6, num_samples=1000, num_features=5):
        X, Y = generate_multi_sine_dataset(seq_len, pred_len, num_samples, num_features)
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


# ======= 训练与验证 =======
def train_multi_feature_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    seq_len, pred_len, feature_dim = 48, 6, 5
    dataset = MultiFeatureDataset(seq_len, pred_len, 2000, feature_dim)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = TimeSeriesTransformer(
        input_features=feature_dim,
        dec_features=feature_dim,
        d_model=128,
        nhead=8,
        num_layers=2,
        output_dim=feature_dim,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    for epoch in range(10):
        model.train()
        total_loss = 0
        for src, tgt in train_loader:
            src, tgt = src.to(device), tgt.to(device)
            decoder_input = torch.zeros_like(tgt)
            decoder_input[:, 1:] = tgt[:, :-1]

            optimizer.zero_grad()
            output = model(src, decoder_input)
            loss = criterion(output, tgt)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}: Loss = {total_loss / len(train_loader):.6f}")

    # ======= 验证 =======
    model.eval()
    x, y = dataset[0]
    x, y = x.unsqueeze(0).to(device), y.unsqueeze(0).to(device)

    pred_in = torch.zeros_like(y)
    for t in range(y.shape[1]):  # 自回归预测
        out = model(x, pred_in)
        pred_in[:, t] = out[:, t].detach()

    pred = pred_in.squeeze(0).cpu().numpy()  # shape (pred_len, feature_dim)
    true = y.squeeze(0).cpu().numpy()

    plt.figure(figsize=(10, 6))
    for f in range(feature_dim):
        plt.subplot(feature_dim, 1, f + 1)
        plt.plot(range(seq_len), x[0, :, f].cpu(), label="Input")
        plt.plot(range(seq_len, seq_len + pred_len), true[:, f], label="True")
        plt.plot(range(seq_len, seq_len + pred_len), pred[:, f], label="Pred")
        plt.legend()
        plt.title(f"Feature {f+1}")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    train_multi_feature_model()
