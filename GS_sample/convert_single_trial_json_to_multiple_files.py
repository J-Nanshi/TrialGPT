#%%
import json
import os
from tqdm import tqdm

#%%
# Load the main JSON file
input_file = "dataset/GS_data/GS_trials_info.json"  # Replace with your actual file name
output_dir = "GS_CT_files/"  # Directory to save the separate JSON files

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Read the JSON data from the input file
with open(input_file, "r") as f:
    data = json.load(f)

# Loop through each NCT_ID entry in the JSON data
for nct_id, details in tqdm(data.items()):
    # Add the NCT_ID as a key-value pair in the details dictionary
    details["NCT_ID"] = nct_id
    
    # Define the output file name
    output_file = os.path.join(output_dir, f"{nct_id}.json")
    
    # Write the individual JSON file
    with open(output_file, "w") as outfile:
        json.dump(details, outfile, indent=4)

    print(f"Saved {output_file}")


print("All files saved successfully!")

# %%
