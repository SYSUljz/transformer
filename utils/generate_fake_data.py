import numpy as np
import pandas as pd
import os

def generate_multi_sine_floods(
    output_dir="./synthetic_floods",
    num_floods=10,         # 洪水事件数量（= CSV 文件数量）
    num_stations=10,       # 测站数（即特征维度）
    seq_len=200,           # 每场洪水的时间步数
    noise_std=0.05,        # 噪声标准差
    max_lag=10,            # 最大时间延迟步数
    random_seed=42,        # 随机种子，保证复现
):
    np.random.seed(random_seed)
    os.makedirs(output_dir, exist_ok=True)

    for flood_id in range(num_floods):
        t = np.linspace(0, 20, seq_len)

        # 生成各测站正弦波信号
        station_data = []
        for s in range(num_stations):
            freq = 0.5 + 0.2 * np.random.rand()         # 不同频率
            phase = np.random.rand() * 2 * np.pi         # 随机相位
            amplitude = 1 + 0.5 * np.random.rand()       # 不同振幅
            signal = amplitude * np.sin(freq * t + phase)
            noise = np.random.normal(0, noise_std, size=seq_len)
            station_data.append(signal + noise)

        station_data = np.stack(station_data, axis=1)  # shape (seq_len, num_stations)

        # ---- 模拟洪水传播的时间延迟 ----
        lags = np.random.randint(0, max_lag + 1, size=num_stations)  # 每个测站的延迟
        delayed_data = np.zeros_like(station_data)

        for i, lag in enumerate(lags):
            if lag > 0:
                delayed_data[lag:, i] = station_data[:-lag, i]  # 右移
            else:
                delayed_data[:, i] = station_data[:, i]

        # ---- 流量为延迟后的加权叠加 ----
        weights = np.random.uniform(0.5, 1.5, size=num_stations)
        flow = delayed_data @ weights / num_stations + np.random.normal(0, noise_std, size=seq_len)

        # 拼接 DataFrame
        df = pd.DataFrame(station_data, columns=[f"station_{i+1}" for i in range(num_stations)])
        for i, lag in enumerate(lags):
            df[f"lag_station_{i+1}"] = delayed_data[:, i]
        df["flow"] = flow

        file_path = os.path.join(output_dir, f"flood_{flood_id:03d}.csv")
        df.to_csv(file_path, index=False)
        print(f"✅ Saved: {file_path} (shape={df.shape}, lags={lags.tolist()})")

    print(f"\n🎯 Done! Generated {num_floods} synthetic floods in: {output_dir}")


if __name__ == "__main__":
    generate_multi_sine_floods(
        output_dir="./processed_data_test",
        num_floods=20,     # 生成20场洪水
        num_stations=10,   # 每场10个测站
        seq_len=300,       # 每场300个时间步
        noise_std=0.05,
        max_lag=2,        # 最大延迟10步
    )
