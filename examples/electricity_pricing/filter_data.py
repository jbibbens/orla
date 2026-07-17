import pandas as pd

# 1. Load data and ensure Time is parsed as a datetime object
df = pd.read_csv("5min_hub_lmps.csv", parse_dates=["Time"])

# 2. Define target locations 
target_locations = [
    "TH_NP15_GEN-APND",  # CAISO example
    "LOUISIANA.HUB",  # MISO example
    "MINN.HUB",          # MISO example
    ".H.INTERNAL_HUB"       # ISONE example
]

# Filter locations
df_filtered = df[df["Location"].isin(target_locations)].copy()

# 3. Find the overlapping times
location_counts = df_filtered.groupby("Time")["Location"].nunique()
valid_timestamps = location_counts[location_counts == len(target_locations)].index
df_intersect = df_filtered[df_filtered["Time"].isin(valid_timestamps)]
df_intersect = df_intersect.sort_values(by=["Time", "Grid", "Location"])

print(f"Filtered down to {df_intersect['Time'].nunique()} overlapping 5-minute intervals.")

# Save
df_intersect.to_csv("filtered_overlapping_hubs.csv", index=False)