import torch
import torch.nn as nn
import copy
from transformer import Encoder, Decoder, EncoderLayer, DecoderLayer, MultiHeadedAttention, PositionwiseFeedForward, PositionalEncoding

class TimeSeriesTransformer(nn.Module):
    """
    A Transformer model adapted for time-series forecasting (regression).

    This model modifies the original Transformer by replacing the embedding layers,
    which are meant for discrete tokens, with linear layers to handle continuous
    input data. It also changes the final output layer to predict a single
    continuous value instead of a vocabulary distribution.
    """
    def __init__(self, input_features, dec_features, d_model=512, N=6, h=8, d_ff=2048, dropout=0.1, max_len=5000):
        """
        Args:
            input_features (int): Number of features in the input sequence (e.g., number of monitoring sites).
            dec_features (int): Number of features in the decoder input sequence (usually 1 for single-variate forecast).
            d_model (int): The dimension of the model's hidden states.
            N (int): The number of encoder and decoder layers.
            h (int): The number of attention heads.
            d_ff (int): The dimension of the feed-forward network.
            dropout (float): The dropout rate.
            max_len (int): The maximum sequence length for positional encoding.
        """
        super().__init__()
        self.d_model = d_model

        # --- Layers for handling continuous data ---
        self.encoder_input_layer = nn.Linear(input_features, d_model)
        self.decoder_input_layer = nn.Linear(dec_features, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len)

        # --- Core Transformer Components (reused from your transformer.py) ---
        attn = MultiHeadedAttention(h, d_model, dropout)
        ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        
        self.encoder = Encoder(EncoderLayer(d_model, copy.deepcopy(attn), copy.deepcopy(ff), dropout), N)
        self.decoder = Decoder(DecoderLayer(d_model, copy.deepcopy(attn), copy.deepcopy(attn), copy.deepcopy(ff), dropout), N)

        # --- Output Layer for Regression ---
        # This layer maps the decoder's output to our desired number of forecast features (1 for 'flow').
        self.output_layer = nn.Linear(d_model, 1)

        # Initialize parameters
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, tgt, src_mask, tgt_mask):
        """
        Forward pass for the time-series Transformer.

        Args:
            src (torch.Tensor): The source sequence. Shape: (batch_size, src_seq_len, input_features).
            tgt (torch.Tensor): The target sequence (for teacher forcing). Shape: (batch_size, tgt_seq_len, dec_features).
            src_mask (torch.Tensor): The source mask.
            tgt_mask (torch.Tensor): The target mask.

        Returns:
            torch.Tensor: The model's predictions. Shape: (batch_size, tgt_seq_len, 1).
        """
        # 1. Project input features to d_model and add positional encoding
        src_embedded = self.positional_encoding(self.encoder_input_layer(src))
        tgt_embedded = self.positional_encoding(self.decoder_input_layer(tgt))

        # 2. Pass through encoder and decoder
        memory = self.encoder(src_embedded, src_mask)
        dec_output = self.decoder(tgt_embedded, memory, src_mask, tgt_mask)

        # 3. Project to final output dimension
        output = self.output_layer(dec_output)
        return output
