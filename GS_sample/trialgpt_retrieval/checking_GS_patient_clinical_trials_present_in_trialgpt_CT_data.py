
#%%
import json

# Load dictionary A from JSON
with open(r"D:\Job\TrialGPT\GS_sample\dataset\trial_info.json", 'r') as f:
    dict_a = json.load(f)
dict_a
#%%
# Load dictionary B from JSONL
dict_b = []
with open(r"D:\Job\TrialGPT\trialgpt_retrieval\synthetic_patient_cases_random_30_modified.jsonl", 'r') as f:
    for line in f:
        dict_b.append(json.loads(line))
dict_b
#%%
# Extract keys from dictionary B and check their presence in dictionary A
missing_keys = []
for entry in dict_b:
    print(entry)
    key_b = entry['_id']
    print(key_b)
    key_a = key_b.split('_')[1] 
    print(key_a) # Extract part after "P_"
    if key_a not in dict_a:
        missing_keys.append(key_b)
#%%
# Print the keys from dictionary B that are not found in dictionary A
if missing_keys:
    print(f"Keys from dictionary B not found in dictionary A: {missing_keys}")
else:
    print("All keys from dictionary B are present in dictionary A.")
# %%
