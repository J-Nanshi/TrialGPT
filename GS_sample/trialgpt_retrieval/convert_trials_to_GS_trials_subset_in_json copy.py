#%%
import json

#importing the clinical trial info processed by trialgpt. it contains all the nctids hopefully
input_json = json.load(open(r"D:\Job\TrialGPT\GS_sample\dataset\trial_info.json"))
#loading the NCTID list we have ingested from hsmd
with open('../../GS_sample/dataset/GS_data/NCT_IDs.txt', 'r') as file:
    # Read the lines into a list
    nct_hsmd_list = file.read().splitlines()

#importing landmark clinical trials
with open('../../GS_sample/dataset/GS_data/landmark_trials 1.json', 'r') as file:
    data = json.load(file)

# Initialize an empty list to store all values
Landmark_trials = []

# Iterate through the dictionary and extend the list with each value
for key, value in data.items():
    Landmark_trials.extend(value)

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

nct_ids_to_process = list(set(nct_hsmd_list + Landmark_trials + patient_nct_ids ))
nct_ids_to_process
# %%
filtered_trial_info = {}

# Iterate through the active trial IDs
for trial_id in nct_ids_to_process:
    # Check if the trial ID exists in the trial_info dictionary
    if trial_id in input_json:
        # Add the trial information to the filtered dictionary
        filtered_trial_info[trial_id] = input_json[trial_id]

# Save the filtered information to a JSON file
output_file = "../../GS_sample/dataset/GS_data/GS_trials_info.json"
with open(output_file, "w") as f:
    json.dump(filtered_trial_info, f, indent=4)

#%%
import json

#importing landmark clinical trials
with open('../../GS_sample/dataset/GS_data/landmark_trials 1.json', 'r') as file:
    data = json.load(file)

# Initialize an empty list to store all values
Landmark_trials = []

# Iterate through the dictionary and extend the list with each value
for key, value in data.items():
    Landmark_trials.extend(value)

Landmark_trials_unique = list(set(Landmark_trials))
Landmark_trials_unique.remove("NCT0464423")
# %%
Landmark_trials_unique
# %%
#Sample code for clinical trial retrival from clinical_trials.gov
import requests
import json

with open('../../GS_sample/dataset/GS_data/landmark_trials 1.json', 'r') as file:
    data = json.load(file)

# Initialize an empty list to store all values
Landmark_trials = []

# Iterate through the dictionary and extend the list with each value
for key, value in data.items():
    Landmark_trials.extend(value)
Landmark_trials_unique = list(set(Landmark_trials))
Landmark_trials_unique.remove("NCT0464423")

diseases = []

for id in Landmark_trials_unique:
    url = f"https://clinicaltrials.gov/api/v2/studies?filter.ids={id}"

    payload = {}
    headers = {}

    response = requests.request("GET", url, headers=headers, data=payload)

    a = response.text
    ctr_json=json.loads(a)
    condition_module = ctr_json["studies"][0]["protocolSection"]["conditionsModule"]["conditions"]
    diseases.append(str(condition_module))
    print(f"{id}: {condition_module}")

# %%
import pandas as pd
df = pd.DataFrame({"NCT_IDS" :Landmark_trials_unique, "Diseases": diseases })
df
# %%
df.to_csv(r"D:\Job\Data\breast_cancer\clinical trial simulation\LTR_filtered\LTR_breast_cancer_filtered_data.csv", index=False)
# %%
