import pandas as pd
import glob
import torch
from torch.utils.data import Dataset
import random


def load_all_floods(data_dir):
    """
    Loads all CSV files from a directory into a list of pandas DataFrames.
    """
    files = glob.glob(f"{data_dir}/*.csv")
    if not files:
        print(f"Warning: No CSV files found in {data_dir}")
    dfs = []
    for f in files:
        df = pd.read_csv(f)
        df["flood_id"] = f.split("/")[-1].replace(".csv", "")
        dfs.append(df)
    return dfs


class FloodDataset(Dataset):
    """
    PyTorch Dataset for flood forecasting. Can be initialized either from a list of
    DataFrames or from a pre-computed list of samples.
    """

    def __init__(
        self, dfs=None, input_window=24, output_window=6, fill_value=0.0, samples=None
    ):
        if samples is not None:
            # Initialize directly from pre-computed samples (for train/val splits)
            self.samples = samples
            print(
                f"✅ Dataset initialized with {len(self.samples)} pre-loaded samples."
            )
            return

        self.samples = []
        if dfs is None:
            raise ValueError("Either 'dfs' or 'samples' must be provided.")

        # Create a unified list of all possible columns across all dataframes
        all_cols = set()
        for df in dfs:
            all_cols.update(df.columns)
        # Ensure 'flow' is the last column for easier indexing later
        self.site_cols = sorted([c for c in all_cols if c.isdigit()]) + ["flow"]

        for df in dfs:
            # Align each dataframe to the full set of columns, filling missing ones
            for c in self.site_cols:
                if c not in df.columns:
                    df[c] = fill_value
            df = df[self.site_cols]

            data = df.fillna(fill_value).values
            n = len(data)
            if n < input_window + output_window:
                continue

            # Create sliding window samples
            for i in range(n - input_window - output_window + 1):
                x = data[i : i + input_window]
                y = data[
                    i + input_window : i + input_window + output_window, -1
                ]  # Target is the 'flow' column
                self.samples.append((x, y))

        print(f"✅ Total samples built: {len(self.samples)} from {len(dfs)} floods")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32),
        )


def create_train_val_datasets(
    dfs, input_window, output_window, fill_value, train_val_split=0.8
):
    """
    Creates and splits the dataset into training and validation sets.
    """
    print(f"Creating full dataset before splitting (split ratio: {train_val_split})...")

    # Create a single dataset instance to generate all possible samples
    full_dataset = FloodDataset(
        dfs=dfs,
        input_window=input_window,
        output_window=output_window,
        fill_value=fill_value,
    )
    all_samples = full_dataset.samples

    # Shuffle and split the generated samples
    random.shuffle(all_samples)
    split_idx = int(len(all_samples) * train_val_split)
    train_samples = all_samples[:split_idx]
    val_samples = all_samples[split_idx:]

    print(
        f"Train samples: {len(train_samples)}, Validation samples: {len(val_samples)}"
    )

    # Create new Dataset objects from the split samples without re-processing
    train_dataset = FloodDataset(samples=train_samples)
    val_dataset = FloodDataset(samples=val_samples)

    return train_dataset, val_dataset
