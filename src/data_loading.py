import os
import fastf1
import pandas as pd


# 1) Setup: caching for reproducibility and faster reloads

# Create a local cache folder (avoids repeated downloads)
os.makedirs('data/cache', exist_ok=True)
fastf1.Cache.enable_cache('data/cache')

# 2024 race names as used by FastF1
gp_names_2024 = [
    "Bahrain", "Saudi Arabia", "Australia", "Japan", "China", "Miami",
    "Emilia Romagna", "Monaco", "Canada", "Spain", "Austria", "Great Britain",
    "Hungary", "Belgium", "Netherlands", "Italy", "Azerbaijan", "Singapore",
    "United States", "Mexico", "Brazil", "Las Vegas", "Qatar", "Abu Dhabi"
]

all_data = []


# 2) Loop through races: load session and extract lap data

for gp in gp_names_2024:
    try:
        print(f"Loading {gp} 2024...")
        # Load the race session for the Grand Prix
        session = fastf1.get_session(2024, gp, 'R')
        session.load()

        # Get lap-by-lap data as a DataFrame
        laps = session.laps.reset_index(drop=True)

        # 3) Event labeling: pit-in (lap the driver ENTERS the pits)

        # PitStopEvent = 1 on the lap the driver enters the pits
        # PitOutTime marks leaving the pits (usually next lap), which we avoid to prevent label shift
        laps['PitStopEvent'] = laps['PitInTime'].notna().astype(int)


        # 4) Feature cleanup: convert LapTime to seconds for modeling

        # FastF1 stores LapTime as a Timedelta -> convert for numeric modeling
        # Some laps may have missing times (e.g., inlaps/outlaps under SC); handle safely
        if 'LapTime' in laps.columns:
            laps['LapTimeSec'] = laps['LapTime'].dt.total_seconds()


        # 5) Stint age feature: laps since last pit OUT (resets after pit-in)

        # We compute StintAge so that the lap immediately AFTER a pit stop has StintAge = 1
        # The pit-in lap retains its current age; the reset occurs on the next lap.
        laps['StintAge'] = 0

        # Process per driver within the session
        for driver in laps['Driver'].unique():
            # Work on a copy to avoid chained assignment confusion
            dmask = laps['Driver'] == driver
            driver_laps = laps.loc[dmask].sort_values('LapNumber')

            current_age = 1
            stint_age_values = []

            # Iterate lap-by-lap in order
            for _, row in driver_laps.iterrows():
                # Assign current stint age to this lap
                stint_age_values.append(current_age)

                # If this lap is a pit-in event, reset age on the next lap
                if row['PitStopEvent'] == 1:
                    current_age = 1
                else:
                    current_age += 1

            # Write back computed ages in the original DataFrame order
            laps.loc[dmask, 'StintAge'] = stint_age_values


        # 6) Optional: include session metadata for grouping later

        # Useful for building sequences per driver per race
        laps['Year'] = 2024
        laps['GrandPrix'] = gp
        laps['SessionType'] = 'R'  # Race

        # Collect data from this session
        all_data.append(laps)

    except Exception as e:
        # Keep the pipeline robust: skip races that fail to load
        print(f"Skipping {gp} 2024 due to error: {e}")


# 7) Combine all races into one dataset

if all_data:
    dataset_2024 = pd.concat(all_data, ignore_index=True)
    print("Final dataset shape:", dataset_2024.shape)
    print(dataset_2024[['Driver', 'LapNumber', 'LapTimeSec', 'Compound', 'StintAge', 'PitStopEvent']].head())
else:
    print("No race data collected.")

# 8) Save/load logic: reproducible artifact on disk

csv_path = "data/f1_2024_laps.csv"

# If a CSV already exists, load it; else save the newly built dataset
if os.path.exists(csv_path):
    dataset_2024 = pd.read_csv(csv_path)
    print("Loaded dataset from CSV")
elif all_data:
    dataset_2024 = pd.concat(all_data, ignore_index=True)
    dataset_2024.to_csv(csv_path, index=False)
    print("Saved dataset to CSV")
else:
    print("No race data collected.")