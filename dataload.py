import pandas as pd
import glob
import torch
from torch.utils.data import Dataset, DataLoader

def load_all_floods(data_dir):
    files = glob.glob(f"{data_dir}/*.csv")
    dfs = []
    for f in files:
        df = pd.read_csv(f)
        df['flood_id'] = f.split('/')[-1].replace('.csv', '')
        dfs.append(df)
    return dfs


class FloodDataset(Dataset):
    def __init__(self, dfs, input_window=24, output_window=6, fill_value=torch.nan):
        self.samples = []

        # Unified list of all site columns (union of all CSVs)
        all_cols = set()
        for df in dfs:
            all_cols |= set(df.columns)
        # Keep only numeric station IDs + 'flow'
        all_cols = sorted([c for c in all_cols if c.isdigit()]) + ['flow']
        self.site_cols = all_cols

        for df in dfs:
            # Align to full set of columns
            for c in self.site_cols:
                if c not in df.columns:
                    df[c] = torch.nan  # mark missing stations as NaN
            df = df[self.site_cols]

            # Replace NaN with 0 (or use another fill strategy)
            data = df.fillna(fill_value).values
            n = len(data)
            if n <= input_window + output_window:
                continue

            for i in range(n - input_window - output_window):
                x = data[i:i + input_window]
                y = data[i + input_window:i + input_window + output_window, -1]
                # mask: 1 where value exists, 0 where originally missing
                mask = ~torch.isnan(torch.tensor(x, dtype=torch.float32))
                self.samples.append((x, y, mask))

        print(f"✅ Total samples built: {len(self.samples)} from {len(dfs)} floods")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y, mask = self.samples[idx]
        return (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32),
            mask
        )


# === Example Usage ===
dfs = load_all_floods("/home/jack_li/python/hydro_data/sorted_data/processed_data")
dataset = FloodDataset(dfs, input_window=24, output_window=6, fill_value=0.0)
if len(dataset) == 0:
    print("⚠️ No valid samples — check your CSVs or flow data.")
else:
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    print("✅ DataLoader ready!")
