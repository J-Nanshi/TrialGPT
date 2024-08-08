#%%
import json
#%%
# Input JSON 
#importing the clinical trial info processed by trialgpt. it contains all the nctids hopefully
input_json = json.load(open(r"D:\Job\TrialGPT\GS_sample\dataset\trial_info.json"))
#%%
#loading the NCTID list we have ingested from hsmd
with open('../../GS_sample/dataset/GS_data/NCT_IDs.txt', 'r') as file:
    # Read the lines into a list
    nct_hsmd_list = file.read().splitlines()
nct_hsmd_list
#%%
#importing landmark clinical trials
with open('../../GS_sample/dataset/GS_data/landmark_trials 1.json', 'r') as file:
    data = json.load(file)

# Initialize an empty list to store all values
Landmark_trials = []

# Iterate through the dictionary and extend the list with each value
for key, value in data.items():
    Landmark_trials.extend(value)

Landmark_trials
#%%
# loading trials form which the synthetic patients were generated
synthetic_patients = []
with open(r"D:\Job\TrialGPT\trialgpt_retrieval\synthetic_patient_cases_random_30_modified.jsonl", 'r') as f:
    for line in f:
        synthetic_patients.append(json.loads(line))
synthetic_patients
patient_nct_ids = []
for entry in synthetic_patients:
    print(entry)
    key_b = entry['_id']
    print(key_b)
    key_a = key_b.split('_')[1] 
    print(key_a)
    patient_nct_ids.append(key_a)

patient_nct_ids
#%%
nct_ids_to_process = list(set(nct_hsmd_list + Landmark_trials + patient_nct_ids ))
nct_ids_to_process
#%%
#creating our own filtered trial cropus jsonl and printing missing keys which are not present in the basic trial info by trialGPT
filtered_json = {key: value for key, value in input_json.items() if key in nct_ids_to_process}
missing_keys = [key for key in nct_ids_to_process if key not in input_json]  #42 not present in TrialGPT
#%%
#to check if there is any missing ncts from all 30 patient details. 
for i in patient_nct_ids:
    if i in missing_keys:
        print(i)
        #NO MISSING NCT FROM SYNTHETIC PATIENT DATA
#%%
# Prepare Output JSONL Line
output = []
for key, value in filtered_json.items():
    entry = {
        "_id": key,
        "title": value["brief_title"],
        "text": f"Summary: {value['brief_summary']}\nInclusion criteria: {value['inclusion_criteria']}\nExclusion criteria: {value['exclusion_criteria']}",
        "metadata": {
            "brief_title": value["brief_title"],
            "phase": value["phase"],
            "drugs": value["drugs"],
            "drugs_list": value["drugs_list"],
            "diseases": value["diseases"],
            "diseases_list": value["diseases_list"],
            "enrollment": value["enrollment"],
            "inclusion_criteria": value["inclusion_criteria"],
            "exclusion_criteria": value["exclusion_criteria"],
            "brief_summary": value["brief_summary"]
        }
    }
    output.append(entry)

# Save the output to JSONL
with open(r'D:\Job\TrialGPT\GS_sample\dataset\GS_data\GS_trials_corpus.jsonl', 'w') as f:
    for entry in output:
        f.write(json.dumps(entry) + '\n')
# %%
output
# %%
