import mlflow
from train_mlflow import train_model

# --- Set up MLflow Experiment ---
# This will create a new experiment if it doesn't exist.
EXPERIMENT_NAME = "Flood Forecasting Transformer"
mlflow.set_experiment(EXPERIMENT_NAME)
print(f"Using MLflow experiment: {EXPERIMENT_NAME}")

# --- Define Base Parameters ---
# These parameters will be common across all runs.
# IMPORTANT: Replace this with the actual path to your data.
DATA_DIRECTORY = "/home/jack_li/python/hydro_data/sorted_data/processed_data"

base_params = {
    "data_dir": DATA_DIRECTORY,
    "model_name": "TimeSeriesTransformer",
    "epochs": 20,
    "batch_size": 32,
    "learning_rate": 1e-4,
    "d_model": 128,
    "n_layers": 3,
    "n_heads": 4,
    "d_ff": 256,
    "dropout": 0.1,
}

# --- Define the Search Space for Hyperparameters ---
# Define different window sizes to test.
input_windows_to_test = [12, 24, 36]
output_windows_to_test = [6, 12]

# --- Run the Experiments ---
print("Starting hyperparameter tuning...")
for input_w in input_windows_to_test:
    for output_w in output_windows_to_test:
        print("-" * 40)
        print(f"Starting run with input_window={input_w}, output_window={output_w}")
        
        # Create a new dictionary for the current run's parameters
        run_params = base_params.copy()
        run_params["input_window"] = input_w
        run_params["output_window"] = output_w

        # The `train_model` function now handles the mlflow.start_run() context
        train_model(run_params)

print("-" * 40)
print("All experiments completed!")
print(f"To view results, run 'mlflow ui' in your terminal and open http://127.0.0.1:5000")
