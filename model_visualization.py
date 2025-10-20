from torchinfo import summary
import torch
from transformer import (
    Transformer,
    CopyDataset,
    DataLoader,
    make_src_mask,
    make_tgt_mask,
    collate_fn,
)
from torch import nn

if __name__ == "__main__":
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
    for epoch in range(1):
        total_loss = 0.0
        for src, tgt_input, tgt in loader:
            src = src.to(device)  # (b, src_len)
            tgt_input = tgt_input.to(device)  # (b, tgt_len)
            tgt = tgt.to(device)  # (b, tgt_len)
            src_mask = make_src_mask(src, pad_token=0).to(device)  # (b,1,1,src_len)
            tgt_mask = make_tgt_mask(tgt_input, pad_token=0).to(
                device
            )  # (b,1,tgt_len,tgt_len)
            summary(model, input_data=(src, tgt_input, src_mask, tgt_mask), depth=4)
            break
