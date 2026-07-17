import pandas as pd
import matplotlib.pyplot as plt
import random

# 1. Update this to your new file's path
file_path = 'filtered_overlapping_hubs.csv'

# Column mappings based on your new file header
time_column = 'Time'
price_column = 'LMP'
location_id_col = 'Location'
location_name_col = 'Location'  # Using name for cleaner legend labels

# How many nodes do you want to see at once?
num_nodes_to_plot = 4

# 2. Read the CSV and automatically parse the EPT column as datetimes
df = pd.read_csv(file_path, parse_dates=[time_column])

# Sort chronologically
df = df.sort_values(time_column)

# 3. Randomly select a subset of nodes based on pnode_id
all_nodes = df[location_id_col].unique()
sample_size = min(num_nodes_to_plot, len(all_nodes))
selected_nodes = random.sample(list(all_nodes), sample_size)

print(f"Plotting data for PNode IDs: {selected_nodes}")

# 4. Create the figure
plt.figure(figsize=(12, 7))

# 5. Loop through and plot each selected node
for node_id in selected_nodes:
    # Filter data for this specific node
    node_data = df[df[location_id_col] == node_id]
    
    # Grab the readable name for the legend (falling back to ID if empty)
    node_name = node_data[location_name_col].iloc[0]
    if pd.isna(node_name) or str(node_name).strip() == "":
        label_text = f"ID: {node_id}"
    else:
        label_text = f"{node_name} ({node_id})"
        
    # Plot using the full datetime on the X-axis
    plt.plot(
        node_data[time_column], 
        node_data[price_column], 
        marker='.', 
        linestyle='-', 
        label=label_text
    )

# 6. Customize Chart
plt.title(f'Total Real-Time LMP Over Time ({sample_size} Random PNodes)', fontsize=14)
plt.xlabel('UTC Start Time', fontsize=12)
plt.ylabel('Total LMP ($/MWh)', fontsize=12)

# Rotates the dates so they display nicely without overlapping
plt.xticks(rotation=45)

# Place the legend neatly outside the chart boundary
plt.legend(title="PNode Name (ID)", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()

plt.show()