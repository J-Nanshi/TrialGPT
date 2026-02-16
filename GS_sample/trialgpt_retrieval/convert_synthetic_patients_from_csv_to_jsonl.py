#%%
import pandas as pd
import json

# Create the DataFrame
df = pd.read_csv("../dataset/GS_data/CTM_evaluation/all_synthetic_cases_active_50cases.csv")
# Convert the DataFrame to JSON Lines format
jsonl_data = df.apply(lambda row: json.dumps({"_id": row["NCT_ID"], "text": row["Synthetic_patients"]}), axis=1)

# Write the JSON Lines to a file
with open(r"..\dataset\GS_data\CTM_evaluation\all_synthetic_cases_active_50cases.jsonl", "w") as f:
    for line in jsonl_data:
        f.write(line + "\n")
# %%
