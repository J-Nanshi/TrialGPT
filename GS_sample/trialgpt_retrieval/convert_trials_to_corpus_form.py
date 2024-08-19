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
# active_100_trials = [
#     "NCT00781612", "NCT01042379", "NCT01785420", "NCT01817452", "NCT01945775", 
#     "NCT02047747", "NCT02062489", "NCT02095184", "NCT02206984", "NCT02264678", 
#     "NCT02344472", "NCT02422641", "NCT02448420", "NCT02476539", "NCT02476786", 
#     "NCT02484404", "NCT02488967", "NCT02535221", "NCT02583828", "NCT02593175", 
#     "NCT02615457", "NCT02627248", "NCT02630368", "NCT02632448", "NCT02641847", 
#     "NCT02694809", "NCT02716116", "NCT02760030", "NCT02810743", "NCT02872025", 
#     "NCT02897700", "NCT02914158", "NCT02926690", "NCT02965950", "NCT02977468", 
#     "NCT03006172", "NCT03025035", "NCT03048942", "NCT03090165", "NCT03095352", 
#     "NCT03150576", "NCT03179904", "NCT03188965", "NCT03213041", "NCT03270007", 
#     "NCT03272334", "NCT03285412", "NCT03306472", "NCT03308201", "NCT03315364", 
#     "NCT03324425", "NCT03326102", "NCT03328026", "NCT03328884", "NCT03344965", 
#     "NCT03351062", "NCT03368729", "NCT03387917", "NCT03401385", "NCT03412643", 
#     "NCT03428802", "NCT03439735", "NCT03444701", "NCT03448042", "NCT03475953", 
#     "NCT03500380", "NCT03504488", "NCT03515798", "NCT03544905", "NCT03546686", 
#     "NCT03561740", "NCT03562637", "NCT03564691", "NCT03564782", "NCT03568656", 
#     "NCT03571633", "NCT03589339", "NCT03596073", "NCT03598257", "NCT03606967", 
#     "NCT03616587", "NCT03620643", "NCT03635632", "NCT03664895", "NCT03671044", 
#     "NCT03674567", "NCT03685331", "NCT03694249", "NCT03709446", "NCT03739931", 
#     "NCT03740256", "NCT03740893", "NCT03742102", "NCT03742245", "NCT03742895", 
#     "NCT03746431", "NCT03747120", "NCT03752398", "NCT03756298"
# ]
#%%
#creating our own filtered trial cropus jsonl and printing missing keys which are not present in the basic trial info by trialGPT
filtered_json = {key: value for key, value in input_json.items() if key in nct_ids_to_process}
missing_keys = [key for key in nct_ids_to_process if key not in input_json]  #42 not present in TrialGPT
#%%


#to check if there is any missing ncts from all 30 patient details. 
for i in nct_ids_to_process:
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
with open(r'..\..\GS_sample\dataset\GS_data\GS_trials_corpus.jsonl', 'w') as f:
    for entry in output:
        f.write(json.dumps(entry) + '\n')
# %%
output
# %%
len(missing_keys)
# %%
