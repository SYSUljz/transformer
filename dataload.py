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
    def __init__(
        self, dfs=None, input_window=24, output_window=6, fill_value=0.0, samples=None
    ):
        if samples is not None:
            self.samples = samples
            print(f"✅ Dataset initialized with {len(self.samples)} pre-loaded samples.")
            return

        if dfs is None:
            raise ValueError("Either 'dfs' or 'samples' must be provided.")

        self.samples = []

        # 获取所有列并统一格式
        all_cols = set()
        for df in dfs:
            all_cols.update(df.columns)

        # ✅ 统一列顺序，确保测站编号列 + flow
        self.site_cols = sorted([c for c in all_cols if str(c).isdigit()]) + ["flow"]

        for df in dfs:
            for c in self.site_cols:
                if c not in df.columns:
                    df[c] = fill_value
            df = df[self.site_cols]

            data = df.fillna(fill_value).values
            n = len(data)
            if n < input_window + output_window:
                continue

            for i in range(n - input_window - output_window + 1):
                x = data[i : i + input_window, :-1]  # ✅ exclude flow
                y = data[i + input_window : i + input_window + output_window, -1]  # only flow
                self.samples.append((x, y))

        print(f"✅ Total samples built: {len(self.samples)} from {len(dfs)} floods")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return (
            torch.tensor(x, dtype=torch.float32),                # (input_window, num_features)
            torch.tensor(y, dtype=torch.float32).unsqueeze(-1),  # (output_window, 1)
        )

def main():
    # === User configuration ===
    data_dir = "/Users/jackli/Downloads/processed_data"       # Folder containing raw flood CSVs
    output_dir = "./dataset"  # Folder to save processed datasets
    input_window = 2                # Input sequence length
    output_window = 1                # Prediction horizon
    fill_value = -1
    train_val_split = 0.8            # Train-validation ratio

    import os
    os.makedirs(output_dir, exist_ok=True)

    print("\n=== Loading all flood data ===")
    dfs = load_all_floods(data_dir)
    if not dfs:
        print("❌ No data found. Please check your directory.")
        return

    print("\n=== Creating train/val datasets ===")
    train_dataset, val_dataset = create_train_val_datasets(
        dfs, input_window, output_window, fill_value, train_val_split
    )

    print(f"\n✅ Train dataset: {len(train_dataset)} samples")
    print(f"✅ Validation dataset: {len(val_dataset)} samples")

    # === Save as PyTorch tensors ===
    torch.save(train_dataset.samples, os.path.join(output_dir, "train_dataset.pt"))
    torch.save(val_dataset.samples, os.path.join(output_dir, "val_dataset.pt"))
    print(f"\n💾 Saved datasets to: {output_dir}/train_dataset.pt and val_dataset.pt")

    # === Optionally export first N samples to CSV for inspection ===
    N = 3  # number of samples to export for quick look
    def samples_to_csv(dataset, name):
        rows = []
        for i in range(min(N, len(dataset))):
            x, y = dataset[i]
            flat_x = x.flatten().tolist()
            rows.append(flat_x + y.tolist())
        cols = [f"x_t{i}" for i in range(x.numel())] + [f"y_t{i}" for i in range(len(y))]
        df = pd.DataFrame(rows, columns=cols)
        csv_path = os.path.join(output_dir, f"{name}_preview.csv")
        df.to_csv(csv_path, index=False)
        print(f"🧾 Saved preview: {csv_path}")

    samples_to_csv(train_dataset, "train")
    samples_to_csv(val_dataset, "val")

    # === Inspect one example ===
    x, y = train_dataset[0]
    print("\n=== Example sample ===")
    print(f"Input shape: {x.shape}, Output shape: {y.shape}")
    print(f"First y values: {y[:5]}")

    # === Example DataLoader usage ===
    from torch.utils.data import DataLoader
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    batch_x, batch_y = next(iter(train_loader))
    print(f"\n🧠 Batch shapes → X: {batch_x.shape}, Y: {batch_y.shape}")
    print("Done ✅")


if __name__ == "__main__":
    main()
