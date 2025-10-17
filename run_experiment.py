import os
import mlflow
from train_mlflow import train_model



# --- 2️⃣ Set MLflow tracking folder inside Drive ---
MLFLOW_DIR = "/home/jack_li/python/hydromodeling/transformer/mlflow_results"
os.makedirs(MLFLOW_DIR, exist_ok=True)
mlflow.set_tracking_uri(f"file://{MLFLOW_DIR}")

# --- 3️⃣ Set up MLflow Experiment ---
EXPERIMENT_NAME = "Flood Forecasting Transformer"
mlflow.set_experiment(EXPERIMENT_NAME)
print(f"[MLflow] Using experiment: {EXPERIMENT_NAME}")
print(f"[MLflow] Tracking URI: {mlflow.get_tracking_uri()}")

# --- 4️⃣ Base Training Parameters ---
DATA_DIRECTORY = "/home/jack_li/python/hydromodeling/transformer/processed_data"
base_params = {
    "data_dir": DATA_DIRECTORY,
    "model_name": "TimeSeriesTransformer",
    "model_name_brief": "TST",
    "epochs": 3,
    "batch_size": 32,
    "learning_rate": 1e-4,
    "d_model": 128,
    "n_layers": 3,
    "n_heads": 4,
    "d_ff": 256,
    "dropout": 0.1,
}

# --- 5️⃣ Hyperparameter Search ---
input_windows_to_test = [24, 36, 48]
output_windows_to_test = [4 ,6, 12]

print("Starting hyperparameter tuning...")
for input_w in input_windows_to_test:
    for output_w in output_windows_to_test:
        print("-" * 40)
        print(f"Starting run: input_window={input_w}, output_window={output_w}")

        run_params = base_params.copy()
        run_params["input_window"] = input_w
        run_params["output_window"] = output_w
        run_params["run_name"] = f"{run_params['model_name_brief']}_input{input_w}_output{output_w}"

        # train_model() should handle mlflow.start_run() internally
        train_model(run_params)

print("-" * 40)
print("✅ All experiments completed!")
print(f"📂 MLflow data saved in: {MLFLOW_DIR}")
