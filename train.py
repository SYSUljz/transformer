import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os

# Import your custom modules
from dataload import load_all_floods, FloodDataset
from forecasting_transformer import TimeSeriesTransformer
from transformer import make_tgt_mask

def train_model(data_dir: str, epochs: int = 10, batch_size: int = 32, model_save_path: str = "flood_model.pth", load_existing_model: bool = False):
    """
    Main function to train the flood forecasting model.

    Args:
        data_dir (str): Path to the directory containing your flood data CSV files.
        epochs (int): Number of training epochs.
        batch_size (int): The batch size for training.
        model_save_path (str): Path to save the trained model.
        load_existing_model (bool): Flag to load a model from model_save_path before training.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # --- 1. Load Data ---
    # NOTE: The dataloader from dataload.py will determine the number of features.
    input_window = 24  # Number of past time steps to use as input
    output_window = 6  # Number of future time steps to predict

    dfs = load_all_floods(data_dir)
    if not dfs:
        print(f"Error: No CSV files found in '{data_dir}'. Please check the path.")
        return

    dataset = FloodDataset(dfs, input_window=input_window, output_window=output_window, fill_value=0.0)
    
    if len(dataset) == 0:
        print("Warning: Dataset is empty. Check if your CSV files have enough data for the specified window sizes.")
        return
        
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Determine feature counts from the dataset
    # x shape: (input_window, num_features), y shape: (output_window,)
    sample_x, sample_y, _ = dataset[0]
    num_input_features = sample_x.shape[1]
    print(f"Detected {num_input_features} input features.")


    # --- 2. Initialize Model ---
    model = TimeSeriesTransformer(
        input_features=num_input_features,
        dec_features=1,  # We predict one feature ('flow') at a time
        d_model=128,     # Smaller model size for potentially smaller dataset
        N=3,             # Fewer layers
        h=4,             # Fewer heads
        d_ff=256,
        dropout=0.1,
        max_len=input_window + output_window
    ).to(device)

    # --- Load existing model if specified ---
    if load_existing_model and os.path.exists(model_save_path):
        print(f"Loading model from {model_save_path}")
        model.load_state_dict(torch.load(model_save_path, map_location=device))
    else:
        print("Initializing a new model.")


    # --- 3. Set up Loss Function and Optimizer ---
    # Use Mean Squared Error for this regression task
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # --- 4. Training Loop ---
    model.train()
    best_loss = float('inf')
    for epoch in range(epochs):
        epoch_loss = 0.0
        for i, (src, tgt_y, src_padding_mask) in enumerate(dataloader):
            # src: [batch, src_len, features]
            # tgt_y: [batch, tgt_len]
            src = src.to(device)
            tgt_y = tgt_y.to(device)
            
            # --- Prepare decoder input (teacher forcing) ---
            # The decoder needs a "shifted right" version of the target sequence.
            # We start with a zero tensor and append the target sequence, removing the last element.
            decoder_input = torch.zeros_like(tgt_y) # Start with zeros
            decoder_input = torch.cat([decoder_input[:, :1], tgt_y[:, :-1]], dim=1)
            decoder_input = decoder_input.unsqueeze(-1).to(device) # Add feature dimension: [batch, tgt_len, 1]

            # --- Create Masks ---
            # For this time-series task, we don't have a special PAD token.
            # The source mask can be all ones (unmasked) if there's no padding.
            # Let's assume no padding for simplicity for now.
            src_mask = torch.ones(src.shape[0], 1, 1, src.shape[1]).to(device) # [batch, 1, 1, src_len]

            # The target mask is crucial to prevent the decoder from seeing future values.
            tgt_mask = make_tgt_mask(tgt_y).to(device) # [batch, 1, tgt_len, tgt_len]

            # --- Forward Pass ---
            optimizer.zero_grad()
            predictions = model(src, decoder_input, src_mask, tgt_mask) # [batch, tgt_len, 1]

            # --- Calculate Loss ---
            # We compare the model's predictions with the actual target values.
            loss = criterion(predictions.squeeze(-1), tgt_y) # Squeeze to match shapes

            # --- Backward Pass ---
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{epochs}], Average Loss: {avg_loss:.6f}")

        # --- Save the model if it has the best loss so far ---
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), model_save_path)
            print(f"Model saved to {model_save_path} with new best loss: {best_loss:.6f}")

if __name__ == "__main__":
    # IMPORTANT: Replace this path with the actual path to your data directory.
    DATA_DIRECTORY = "/home/jack_li/python/hydro_data/sorted_data/processed_data"
    MODEL_SAVE_PATH = "flood_forecaster_best.pth"

    # Set load_existing_model to True if you want to resume training from a checkpoint
    train_model(
        data_dir=DATA_DIRECTORY, 
        epochs=20, 
        batch_size=32,
        model_save_path=MODEL_SAVE_PATH,
        load_existing_model=False 
    )

