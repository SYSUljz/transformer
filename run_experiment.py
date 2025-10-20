import os
import mlflow
from train_mlflow import train_model


# --- 2️⃣ Set MLflow tracking folder inside Drive ---
MLFLOW_DIR = "/Users/jackli/Desktop/python/transformer/mlflow_results"
os.makedirs(MLFLOW_DIR, exist_ok=True)
mlflow.set_tracking_uri(f"file://{MLFLOW_DIR}")

# --- 3️⃣ Set up MLflow Experiment ---
EXPERIMENT_NAME = "Flood Forecasting Transformer"
mlflow.set_experiment(EXPERIMENT_NAME)
print(f"[MLflow] Using experiment: {EXPERIMENT_NAME}")
print(f"[MLflow] Tracking URI: {mlflow.get_tracking_uri()}")

# --- 4️⃣ Base Training Parameters ---
DATA_DIRECTORY = "/Users/jackli/Downloads/processed_data"
# File: run_experiment.py

# --- 4️⃣ Base Training Parameters ---
DATA_DIRECTORY = "/Users/jackli/Downloads/processed_data"
# ✅ Define the separate directory for prediction data
PREDICT_DATA_DIRECTORY = "/Users/jackli/Desktop/python/transformer/predict_data"

base_params = {
    "data_dir": DATA_DIRECTORY,
    "predict_data_dir": PREDICT_DATA_DIRECTORY,  # Add this line
    "model_name": "TimeSeriesTransformer",
    "model_name_brief": "TST",
    "epochs": 20,
    "batch_size": 32,
    "learning_rate": 4*1e-4,
    "d_model": 128,
    "n_layers": 3,
    "n_heads": 4,
    "d_ff": 256,
    "dropout": 0.1,
}

# ... (the rest of the file remains the same)

# --- 5️⃣ Hyperparameter Search ---5
# input_windows_to_test = [24, 36, 48]
# output_windows_to_test = [4 ,6, 12]
input_windows_to_test = [48]
output_windows_to_test = [4]

print("Starting hyperparameter tuning...")
for input_w in input_windows_to_test:
    for output_w in output_windows_to_test:
        print("-" * 40)
        print(f"Starting run: input_window={input_w}, output_window={output_w}")

        run_params = base_params.copy()
        run_params["input_window"] = input_w
        run_params["output_window"] = output_w
        run_params["run_name"] = (
            f"{run_params['model_name_brief']}_input{input_w}_output{output_w}"
        )

        # train_model() should handle mlflow.start_run() internally
        train_model(run_params)

print("-" * 40)
print("✅ All experiments completed!")
print(f"📂 MLflow data saved in: {MLFLOW_DIR}")
