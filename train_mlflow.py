import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import mlflow
import mlflow.pytorch

# Import your custom modules
from dataload import load_all_floods, FloodDataset
from forecasting_transformer import TimeSeriesTransformer
from transformer import make_tgt_mask

def train_model(params: dict):
    """
    Main function to train the flood forecasting model with MLflow tracking.

    Args:
        params (dict): A dictionary containing all hyperparameters.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # --- Start an MLflow Run ---
    with mlflow.start_run():
        print("Starting MLflow run...")
        # Log parameters from the params dictionary
        mlflow.log_params(params)
        print(f"Logged parameters: {params}")

        # --- 1. Load Data ---
        print("Loading data...")
        dfs = load_all_floods(params["data_dir"])
        if not dfs:
            print(f"Error: No CSV files found in '{params['data_dir']}'.")
            return

        dataset = FloodDataset(dfs, 
                               input_window=params["input_window"], 
                               output_window=params["output_window"], 
                               fill_value=0.0)
        
        if len(dataset) == 0:
            print("Warning: Dataset is empty for the given window sizes.")
            return
            
        dataloader = DataLoader(dataset, batch_size=params["batch_size"], shuffle=True)
        
        sample_x, _, _ = dataset[0]
        num_input_features = sample_x.shape[1]
        print(f"Detected {num_input_features} input features.")


        # --- 2. Initialize Model ---
        # Here you could add logic to select different models based on params['model_name']
        print(f"Initializing model: {params.get('model_name', 'TimeSeriesTransformer')}")
        model = TimeSeriesTransformer(
            input_features=num_input_features,
            dec_features=1,
            d_model=params["d_model"],
            N=params["n_layers"],
            h=params["n_heads"],
            d_ff=params["d_ff"],
            dropout=params["dropout"],
            max_len=params["input_window"] + params["output_window"]
        ).to(device)

        # --- 3. Set up Loss Function and Optimizer ---
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=params["learning_rate"])

        # --- 4. Training Loop ---
        model.train()
        best_loss = float('inf')
        print("Starting training loop...")
        for epoch in range(params["epochs"]):
            epoch_loss = 0.0
            for i, (src, tgt_y, _) in enumerate(dataloader):
                src = src.to(device)
                tgt_y = tgt_y.to(device)
                
                decoder_input = torch.zeros_like(tgt_y)
                decoder_input = torch.cat([decoder_input[:, :1], tgt_y[:, :-1]], dim=1)
                decoder_input = decoder_input.unsqueeze(-1).to(device)

                src_mask = torch.ones(src.shape[0], 1, 1, src.shape[1]).to(device)
                tgt_mask = make_tgt_mask(tgt_y).to(device)

                optimizer.zero_grad()
                predictions = model(src, decoder_input, src_mask, tgt_mask)
                loss = criterion(predictions.squeeze(-1), tgt_y)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(dataloader)
            print(f"Epoch [{epoch+1}/{params['epochs']}], Average Loss: {avg_loss:.6f}")
            
            # Log metrics to MLflow
            mlflow.log_metric("avg_loss", avg_loss, step=epoch)

            # --- Save the model artifact if it has the best loss so far ---
            if avg_loss < best_loss:
                best_loss = avg_loss
                mlflow.log_metric("best_loss", best_loss, step=epoch)
                # Log the model using MLflow's PyTorch integration
                mlflow.pytorch.log_model(model, "best_model")
                print(f"New best model logged to MLflow with loss: {best_loss:.6f}")
        
        print("MLflow run finished.")
