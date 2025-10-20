# transformer_scratch.py
import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


# -------------------------
# Helper: clones
# -------------------------
def clones(module, N):
    "深拷贝 N 份相同模块（用于堆叠层）"
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


# -------------------------
# Positional Encoding (sinusoidal)
# -------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)  # (max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()  # (max_len,1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)  # even
        pe[:, 1::2] = torch.cos(position * div_term)  # odd
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)  # 非可训练参数

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :].to(x.device)


# -------------------------
# Scaled Dot-Product Attention
# -------------------------
def attention(query, key, value, mask=None, dropout=None):
    # query/key/value: (batch, heads, seq_len, d_k)
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(
        d_k
    )  # (..., seq_q, seq_k)
    if mask is not None:
        # mask broadcastable to scores; masked positions set to very negative
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


# -------------------------
# Multi-Head Attention
# -------------------------
class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super().__init__()
        assert d_model % h == 0
        self.h = h
        self.d_k = d_model // h
        self.linears = clones(nn.Linear(d_model, d_model), 4)  # q,k,v, output
        self.attn = None
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        # query/key/value: (batch, seq_len, d_model)
        nbatches = query.size(0)

        # 1) linear projections and split heads
        def reshape_linear(x, linear):
            x = linear(x)  # (batch, seq_len, d_model)
            x = x.view(nbatches, -1, self.h, self.d_k).transpose(
                1, 2
            )  # (batch, h, seq_len, d_k)
            return x

        query, key, value = [
            reshape_linear(x, lin)
            for x, lin in zip((query, key, value), self.linears[:3])
        ]
        # 2) apply attention on all heads
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)
        # 3) concat heads
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        # 4) final linear
        return self.linears[-1](x)  # (batch, seq_len, d_model)


# -------------------------
# Position-wise Feed-Forward
# -------------------------
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w2(self.dropout(F.relu(self.w1(x))))


# -------------------------
# Encoder Layer
# -------------------------
class EncoderLayer(nn.Module):
    def __init__(self, d_model, self_attn, feed_forward, dropout):
        super().__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(
            nn.ModuleList([nn.Identity(), nn.Identity()]), 2
        )  # placeholders
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, src_mask):
        # x: (batch, seq, d_model)
        # self-attention + add & norm
        x2 = self.self_attn(x, x, x, mask=src_mask)
        x = self.norm1(x + self.dropout(x2))
        # feed forward + add & norm
        x2 = self.feed_forward(x)
        x = self.norm2(x + self.dropout(x2))
        return x


# -------------------------
# Encoder (stack of N layers)
# -------------------------
class Encoder(nn.Module):
    def __init__(self, layer, N):
        super().__init__()
        self.layers = clones(layer, N)
        self.norm = nn.LayerNorm(layer.self_attn.linears[0].in_features)  # d_model

    def forward(self, x, src_mask):
        for layer in self.layers:
            x = layer(x, src_mask)
        return self.norm(x)


# -------------------------
# Decoder Layer
# -------------------------
class DecoderLayer(nn.Module):
    def __init__(self, d_model, self_attn, src_attn, feed_forward, dropout):
        super().__init__()
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, memory, src_mask, tgt_mask):
        # x: (batch, tgt_seq, d_model)
        # 1) masked self-attn
        x2 = self.self_attn(x, x, x, mask=tgt_mask)
        x = self.norm1(x + self.dropout(x2))
        # 2) encoder-decoder attention
        x2 = self.src_attn(x, memory, memory, mask=src_mask)
        x = self.norm2(x + self.dropout(x2))
        # 3) feed forward
        x2 = self.feed_forward(x)
        x = self.norm3(x + self.dropout(x2))
        return x


# -------------------------
# Decoder (stack of N layers)
# -------------------------
class Decoder(nn.Module):
    def __init__(self, layer, N):
        super().__init__()
        self.layers = clones(layer, N)
        self.norm = nn.LayerNorm(layer.self_attn.linears[0].in_features)  # d_model

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


# -------------------------
# Full Transformer (encoder-decoder)
# -------------------------
class Transformer(nn.Module):
    def __init__(
        self,
        src_vocab,
        tgt_vocab,
        d_model=512,
        N=6,
        h=8,
        d_ff=2048,
        dropout=0.1,
        max_len=5000,
    ):
        super().__init__()
        self.d_model = d_model
        # Embeddings
        self.src_embed = nn.Sequential(
            nn.Embedding(src_vocab, d_model), PositionalEncoding(d_model, max_len)
        )
        self.tgt_embed = nn.Sequential(
            nn.Embedding(tgt_vocab, d_model), PositionalEncoding(d_model, max_len)
        )

        # Build attention and feed-forward
        attn = MultiHeadedAttention(h, d_model, dropout)
        ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        # Encoder / Decoder stacks
        self.encoder = Encoder(
            EncoderLayer(d_model, copy.deepcopy(attn), copy.deepcopy(ff), dropout), N
        )
        self.decoder = Decoder(
            DecoderLayer(
                d_model,
                copy.deepcopy(attn),
                copy.deepcopy(attn),
                copy.deepcopy(ff),
                dropout,
            ),
            N,
        )

        # Generator (to vocab)
        self.generator = nn.Linear(d_model, tgt_vocab)

        # initialize parameters (same as original)
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, tgt, memory, src_mask, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)

    def forward(self, src, tgt, src_mask, tgt_mask):
        "src: (b, src_len), tgt: (b, tgt_len)"
        memory = self.encode(src, src_mask)
        dec = self.decode(tgt, memory, src_mask, tgt_mask)
        out = self.generator(dec)  # logits (b, tgt_len, tgt_vocab)
        return out


# -------------------------
# Masks helpers
# -------------------------
def make_padding_mask(seq, pad_token=0):
    # seq: (batch, seq_len), returns mask (batch, seq_len) where 1 means token present
    mask = (seq != pad_token).type(torch.uint8)
    return mask  # dtype uint8 for masked_fill (0 -> masked)


def make_src_mask(src, pad_token=0):
    # for encoder-decoder attention we need shape (batch, 1, 1, src_len) or broadcastable
    mask = (
        make_padding_mask(src, pad_token).unsqueeze(1).unsqueeze(1)
    )  # (b,1,1,src_len)
    return mask


def make_tgt_mask(tgt, pad_token=0):
    # combine padding mask and subsequent mask
    batch_size, tgt_len = tgt.size()
    # padding mask: (b, 1, 1, tgt_len)
    pad_mask = make_padding_mask(tgt, pad_token).unsqueeze(1).unsqueeze(1)
    # subsequent mask: (1, 1, tgt_len, tgt_len)
    subsequent = torch.triu(
        torch.ones((tgt_len, tgt_len), device=tgt.device), diagonal=1
    ).type(torch.uint8)
    subsequent_mask = (subsequent == 0).unsqueeze(0).unsqueeze(0)  # True where allowed
    # combine: allowed if both pad_mask==1 and subsequent_mask==1
    combined = pad_mask & subsequent_mask
    return combined  # boolean mask (b,1,tgt_len,tgt_len)


# -------------------------
# Toy dataset: copy task
# -------------------------
class CopyDataset(Dataset):
    def __init__(self, n_samples=10000, seq_len=10, vocab_size=11):
        # use tokens 1..vocab_size-1 as real tokens, 0 as PAD, and let's define 1 as START too maybe
        self.data = []
        for _ in range(n_samples):
            length = seq_len
            seq = torch.randint(2, vocab_size, (length,))  # avoid 0/1
            self.data.append(seq)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        src = self.data[idx]  # (seq_len,)
        tgt = self.data[idx]
        # for simple copy task, target input is shifted right with start token 1
        start_token = 1
        tgt_input = torch.cat([torch.tensor([start_token]), tgt[:-1]])
        return src, tgt_input, tgt  # note: tgt is the expected output


def collate_fn(batch):
    srcs, tgt_inputs, tgts = zip(*batch)
    srcs = torch.stack(srcs)
    tgt_inputs = torch.stack(tgt_inputs)
    tgts = torch.stack(tgts)
    return srcs, tgt_inputs, tgts


# -------------------------
# Greedy decode (inference)
# -------------------------
def greedy_decode(model, src, src_mask, max_len, start_symbol=1, device="cpu"):
    # src: (1, src_len)
    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type_as(src).to(device)  # (1,1)
    for i in range(max_len):
        tgt_mask = make_tgt_mask(ys, pad_token=0).to(device)  # (1,1,cur_len,cur_len)
        out = model.decode(ys, memory, src_mask, tgt_mask)  # (1, cur_len, d_model)
        prob = model.generator(out[:, -1])  # last token logits (1, vocab)
        next_word = torch.argmax(prob, dim=-1).unsqueeze(0)  # (1,1)
        ys = torch.cat([ys, next_word], dim=1)
    return ys  # (1, max_len+1)


# -------------------------
# Small training loop example
# -------------------------
def run_toy_training():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab_size = 12  # 0: PAD, 1: START, 2..11 tokens
    model = Transformer(
        src_vocab=vocab_size,
        tgt_vocab=vocab_size,
        d_model=128,
        N=2,
        h=4,
        d_ff=256,
        dropout=0.1,
        max_len=50,
    ).to(device)

    dataset = CopyDataset(n_samples=2000, seq_len=8, vocab_size=vocab_size)
    loader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)

    criterion = nn.CrossEntropyLoss(ignore_index=0)  # ignore pad if present
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    for epoch in range(5):
        total_loss = 0.0
        for src, tgt_input, tgt in loader:
            src = src.to(device)  # (b, src_len)
            tgt_input = tgt_input.to(device)  # (b, tgt_len)
            tgt = tgt.to(device)  # (b, tgt_len)
            src_mask = make_src_mask(src, pad_token=0).to(device)  # (b,1,1,src_len)
            tgt_mask = make_tgt_mask(tgt_input, pad_token=0).to(
                device
            )  # (b,1,tgt_len,tgt_len)

            out = model(src, tgt_input, src_mask, tgt_mask)  # (b, tgt_len, vocab)
            loss = criterion(out.view(-1, out.size(-1)), tgt.view(-1))
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
        print(f"Epoch {epoch+1}, avg loss: {total_loss / len(loader):.4f}")

    # test greedy decode on a random example
    model.eval()
    src_example = dataset[0][0].unsqueeze(0).to(device)  # (1, src_len)
    src_mask = make_src_mask(src_example, pad_token=0).to(device)
    decoded = greedy_decode(
        model, src_example, src_mask, max_len=8, start_symbol=1, device=device
    )
    print("src:", src_example.cpu().numpy())
    print("decoded:", decoded.cpu().numpy())


if __name__ == "__main__":
    run_toy_training()
