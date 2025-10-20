import torch
import os
import argparse
import mlflow
import pandas as pd
from torch.utils.data import DataLoader

# Import your custom modules
from dataload import load_all_floods, FloodDataset
from forecasting_transformer import TimeSeriesTransformer
from transformer import make_tgt_mask


def load_model_from_run(run_id: str):
    """
    Loads a trained PyTorch model directly from an MLflow run.

    Args:
        run_id (str): The unique ID of the MLflow run to load from.

    Returns:
        tuple: A tuple containing:
            - model: The loaded PyTorch model.
            - params (dict): A dictionary of the hyperparameters used for the run.
    """
    print(f"Loading model from MLflow run ID: {run_id}")
    logged_model = f"runs:/{run_id}/best_model"
    model = mlflow.pytorch.load_model(logged_model)

    # Load parameters from the client
    client = mlflow.tracking.MlflowClient()
    run_data = client.get_run(run_id).data
    params = run_data.params

    # Convert numeric params to int or float
    for key, value in params.items():
        try:
            if "." in value:
                params[key] = float(value)
            else:
                params[key] = int(value)
        except ValueError:
            pass  # Keep as string if conversion fails

    print("✅ Model and parameters loaded successfully.")
    return model, params


def predict(model, dataloader, params, device):
    """
    Generates predictions for a given dataset using an auto-regressive approach.
    """
    model.eval()
    all_preds = []
    all_reals = []
    output_window = params["output_window"]

    with torch.no_grad():
        for i, (src, tgt_y) in enumerate(dataloader):
            src, tgt_y = src.to(device), tgt_y.to(device)

            memory = model.encoder(
                model.positional_encoding(model.encoder_input_layer(src))
            )
            decoder_input = torch.zeros(src.shape[0], 1, 1).to(device)

            for _ in range(output_window):
                tgt_mask = make_tgt_mask(decoder_input.squeeze(-1)).to(device)
                tgt_embedded = model.positional_encoding(
                    model.decoder_input_layer(decoder_input)
                )
                dec_output = model.decoder(tgt_embedded, memory, tgt_mask=tgt_mask)
                output = model.output_layer(dec_output[:, -1:, :])
                decoder_input = torch.cat([decoder_input, output], dim=1)

            final_preds = decoder_input[:, 1:, :].squeeze(-1)
            all_preds.append(final_preds.cpu())
            all_reals.append(tgt_y.cpu())

    return torch.cat(all_preds), torch.cat(all_reals)


def main():
    parser = argparse.ArgumentParser(
        description="Predict using a model from an MLflow run."
    )
    parser.add_argument(
        "--run_id", type=str, required=True, help="The ID of the MLflow run."
    )
    parser.add_argument(
        "--predict_data_dir",
        type=str,
        required=True,
        help="Directory with new data for prediction.",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="predictions.csv",
        help="Path to save the output CSV.",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load the model and its parameters from MLflow
    model, params = load_model_from_run(args.run_id)
    model.to(device)

    # 2. Load the new dataset for prediction
    dfs = load_all_floods(args.predict_data_dir)
    dataset = FloodDataset(
        dfs,
        input_window=params["input_window"],
        output_window=params["output_window"],
        fill_value=0.0,
    )
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

    # 3. Generate predictions
    predictions, ground_truth = predict(model, dataloader, params, device)

    # 4. Save results to a CSV file
    pred_df = pd.DataFrame(
        predictions.numpy(),
        columns=[f"pred_step_{i+1}" for i in range(predictions.shape[1])],
    )
    truth_df = pd.DataFrame(
        ground_truth.numpy(),
        columns=[f"true_step_{i+1}" for i in range(ground_truth.shape[1])],
    )
    results_df = pd.concat([truth_df, pred_df], axis=1)
    results_df.to_csv(args.output_csv, index=False)

    print(f"✅ Successfully saved predictions to {args.output_csv}.")


if __name__ == "__main__":
    main()
