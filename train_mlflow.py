# File: train_mlflow.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import mlflow
import mlflow.pytorch
import pandas as pd # ✅ Import pandas

# Import your custom modules
from dataload import load_all_floods, FloodDataset
from transformer_torch import TimeSeriesTransformer


# 💡 NEW HELPER FUNCTION: To generate predictions and save them
def validate_and_save_predictions(model, params, device, epoch):
    """
    Generates and saves predictions on the validation dataset.
    This uses the same auto-regressive logic as your predict.py.
    """
    print("\n--- Running validation and saving predictions ---")
    predict_data_dir = params.get("predict_data_dir")
    if not predict_data_dir:
        print("Warning: 'predict_data_dir' not in params. Skipping validation.")
        return

    # 1. Load validation data
    dfs = load_all_floods(predict_data_dir)
    if not dfs:
        print(f"Warning: No CSV files found in prediction dir '{predict_data_dir}'.")
        return
        
    dataset = FloodDataset(
        dfs,
        input_window=params["input_window"],
        output_window=params["output_window"],
        fill_value=0.0
    )
    if len(dataset) == 0:
        print("Warning: Prediction dataset is empty.")
        return
        
    dataloader = DataLoader(dataset, batch_size=params["batch_size"], shuffle=False)
    
    # 2. Generate predictions
    model.eval() # Set model to evaluation mode
    all_preds, all_reals = [], []
    output_window = params['output_window']

    with torch.no_grad():
        for src, tgt_y in dataloader:
            src, tgt_y = src.to(device), tgt_y.to(device)
            
            # Create masks and initial decoder input for a single forward pass
            src_mask = torch.ones(src.shape[0], 1, 1, src.shape[1]).to(device)
            
            # Initialize decoder_input with zeros for the entire output window
            # Shape: [batch_size, output_window, 1]
            decoder_input = torch.zeros(src.shape[0], output_window, 1).to(device)
            
            # Create target mask for the decoder
            # Squeeze to remove the last dimension for mask creation: [batch_size, output_window]
            #tgt_mask = make_tgt_mask(decoder_input.squeeze(-1)).to(device)
            
            # Generate predictions in a single forward pass as requested
            predictions = model(src, decoder_input, src_mask, tgt_mask)
            
            # Collect results
            final_preds = predictions.squeeze(-1) # Squeeze the feature dimension
            all_preds.append(final_preds.cpu())
            all_reals.append(tgt_y.cpu())
            
    predictions = torch.cat(all_preds)
    ground_truth = torch.cat(all_reals)

    # 3. Save results to a CSV
    pred_df = pd.DataFrame(predictions.numpy(), columns=[f'pred_step_{i+1}' for i in range(predictions.shape[1])])
    truth_df = pd.DataFrame(ground_truth.numpy(), columns=[f'true_step_{i+1}' for i in range(ground_truth.shape[1])])
    results_df = pd.concat([truth_df, pred_df], axis=1)
    
    # Save locally first
    output_dir = "predictions"
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, f"predictions_epoch_{epoch+1}.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"✅ Saved validation predictions to {csv_path}")

    # 4. Log the CSV to MLflow
    mlflow.log_artifact(csv_path)
    print("✅ Logged prediction CSV as an MLflow artifact.")
    
    model.train() # Set model back to training mode
    print("--- Finished validation ---\n")

def train_model(params: dict):
    """
    Main function to train the flood forecasting model with MLflow tracking.

    Args:
        params (dict): A dictionary containing all hyperparameters.
    """
    device = torch.device('mps' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # --- Start an MLflow Run ---
    with mlflow.start_run(run_name=params.get("run_name", "default_run")):
        print("Starting MLflow run...")
        mlflow.log_params(params)

        # --- 1. Load Data ---
        print("Loading data...")
        dfs = load_all_floods(params["data_dir"])
        if not dfs:
            print(f"Error: No CSV files found in '{params['data_dir']}'.")
            return

        dataset = FloodDataset(
            dfs, 
            input_window=params["input_window"], 
            output_window=params["output_window"], 
            fill_value=0.0
        )

        if len(dataset) == 0:
            print("Warning: Dataset is empty for the given window sizes.")
            return

        dataloader = DataLoader(dataset, batch_size=params["batch_size"], shuffle=True)

        sample_x, _ = dataset[0]
        num_input_features = sample_x.shape[1]
        print(f"Detected {num_input_features} input features.")

        # --- 2. Initialize Model ---
        print(f"Initializing model: {params.get('model_name', 'TimeSeriesTransformer')}")
        model = TimeSeriesTransformer(
            input_features=num_input_features,
            dec_features=1,
            d_model=params["d_model"],
            num_layers=params["n_layers"],
            nhead=params["n_heads"],
            dim_feedforward=params["d_ff"],
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

        for epoch in range(60):
            model.train()
            total_loss = 0
            for src, tgt in dataloader:
                src, tgt = src.to(device), tgt.to(device)
                decoder_input = torch.zeros_like(tgt)
                decoder_input[:, 1:] = tgt[:, :-1]

                optimizer.zero_grad()
                output = model(src, decoder_input)
                loss = criterion(output, tgt)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f"Epoch {epoch+1}: Loss = {total_loss / len(dataloader):.6f}")
            # --- Save best model ---
            # if avg_loss < best_loss:
            #     best_loss = avg_loss
            #     mlflow.log_metric("best_loss", best_loss, step=epoch)

            #     # ✅ Save the .pth file locally
            #     save_path = os.path.join("checkpoints", "best_model.pth")
            #     os.makedirs(os.path.dirname(save_path), exist_ok=True)
            #     torch.save(model.state_dict(), save_path)
            #     print(f"Best model saved locally: {save_path}")

            #     # ✅ Log the .pth file to MLflow as artifact
            #     mlflow.log_artifact(save_path)

            #     # ✅ Also log full model to MLflow (with environment info)
            #     mlflow.pytorch.log_model(model, "best_model")
                
            #     print(f"New best model logged to MLflow with loss: {best_loss:.6f}")
                
            #     # 💡 CALL THE NEW VALIDATION FUNCTION HERE
            #     # This will run inference on your separate data and save the CSV.
            #     #validate_and_save_predictions(model, params, device, epoch)

        print("MLflow run finished.")

