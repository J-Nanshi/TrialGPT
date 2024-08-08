#%%
import json

# Input data
with open(r'..\..\GS_sample\dataset\GS_data\retrieval_keywords_gpt4turbo_synthetic_patient_cases_random_30.json', 'r') as f:
    keyword_retrieval = json.load(f)
keyword_retrieval

#%%
# Load synthetic_patient.jsonl content
synthetic_patient_data = []
with open(r"..\..\GS_sample\dataset\GS_data\synthetic_patient_cases_random_30_modified.jsonl", 'r') as f:
    for line in f:
        synthetic_patient_data.append(json.loads(line))
synthetic_patient_data
#%%
# Convert synthetic_patient.jsonl data to a dictionary
id_to_text = {entry["_id"]: entry["text"] for entry in synthetic_patient_data}
id_to_text
#%%
# Construct the id2queries dictionary
id2queries = {}

for key, value in keyword_retrieval.items():
    if key in id_to_text:
        id2queries[key] = {
            "raw": id_to_text[key],  # This is a simplification; adjust as needed for the full text
            "gpt-4-turbo": {
                "summary": value["summary"],
                "conditions": value["conditions"]
            }
        }

id2queries

# Print the final id2queries dictionary
with open(r'..\..\GS_sample\dataset\GS_data\GS_id2quries_all_30_synthetic_patients.json', 'w') as f:
    json.dump(id2queries, f, indent=4)

# %%
