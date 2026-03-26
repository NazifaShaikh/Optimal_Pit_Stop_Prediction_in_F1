import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
import joblib

# 1) Load dataset

dataset_2024 = pd.read_csv("data/f1_2024_laps.csv")
dataset_2024.columns = dataset_2024.columns.str.strip()  # clean column names

# 2) Ensure LapTime is numeric (seconds)

if 'LapTimeSeconds' not in dataset_2024.columns:
    dataset_2024['LapTime'] = pd.to_timedelta(dataset_2024['LapTime'])
    dataset_2024['LapTimeSeconds'] = dataset_2024['LapTime'].dt.total_seconds()

# 3) Handle missing values

for col in ['LapTimeSeconds', 'StintAge']:
    if dataset_2024[col].isnull().any():
        dataset_2024[col] = dataset_2024[col].fillna(dataset_2024[col].median())

# 4) Encode tyre compound

dataset_2024 = pd.get_dummies(dataset_2024, columns=['Compound'], prefix='Tyre')

# 5) Normalize numerical features (RobustScaler handles outliers better)

scaler = RobustScaler()
dataset_2024[['LapTimeSeconds', 'StintAge']] = scaler.fit_transform(
    dataset_2024[['LapTimeSeconds', 'StintAge']]
)
joblib.dump(scaler, "data/scaler.pkl")  # save scaler for inference

# 6) Label engineering: laps until next pit (capped horizon)

dataset_2024['LapsUntilPit'] = np.nan
max_horizon = 20

for (gp, driver), group in dataset_2024.groupby(['GrandPrix', 'Driver']):
    pit_laps = group[group['PitStopEvent'] == 1]['LapNumber'].tolist()
    pit_laps.sort()

    next_pit_map = {}
    for lap_num in group['LapNumber']:
        future_pits = [p for p in pit_laps if p > lap_num]
        next_pit_map[lap_num] = (
            min(future_pits[0] - lap_num, max_horizon) if future_pits else max_horizon
        )

    dataset_2024.loc[
        (dataset_2024['GrandPrix'] == gp) & (dataset_2024['Driver'] == driver),
        'LapsUntilPit'
    ] = dataset_2024.loc[
        (dataset_2024['GrandPrix'] == gp) & (dataset_2024['Driver'] == driver),
        'LapNumber'
    ].map(next_pit_map)

# 7) Build GRU sequences

def build_sequences(df, seq_len=15):
    X, y = [], []
    features = [
        'LapTimeSeconds', 'StintAge',
        'Tyre_SOFT', 'Tyre_MEDIUM', 'Tyre_HARD'
    ]

    for (gp, driver), group in df.groupby(['GrandPrix', 'Driver']):
        driver_laps = group.reset_index(drop=True)

        for i in range(len(driver_laps) - seq_len):
            seq = driver_laps.loc[i:i + seq_len - 1, features].values
            target = driver_laps.loc[i + seq_len - 1, 'LapsUntilPit']
            if not np.isnan(target):
                X.append(seq)
                y.append(target)

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

X, y = build_sequences(dataset_2024, seq_len=15)

print("X shape:", X.shape)  # (num_sequences, 15, num_features)
print("y shape:", y.shape)  # (num_sequences,)

np.savez_compressed("data/f1_sequences.npz", X=X, y=y)