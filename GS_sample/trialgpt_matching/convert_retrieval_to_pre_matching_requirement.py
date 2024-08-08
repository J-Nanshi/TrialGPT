#%%
import json

# Loading the output of retrieval process
with open('../../GS_sample/results/qid2nctids_all_30_patients_results_GS_data_k50_bm25wt1_medcptwt1_N700.json') as f:
    retrived_ncts = json.load(f)
retrived_ncts
#%%
#loading the synthetic patient raw detials
with open('../../GS_sample/dataset/GS_data/synthetic_patient_cases_random_30_modified.jsonl') as f:
    patient_queries = [json.loads(line) for line in f]
patient_queries
#%%
# loading TrialGPT clinical trial info
with open('../../GS_sample/dataset/trial_info.json') as f:
    trial_info = json.load(f)
trial_info
#%%
# Process patient queries into a dictionary
patient_dict = {item['_id']: item['text'] for item in patient_queries}
patient_dict
#%%
# Initialize the output list
output = []

# Iterate over the retrieved NCTs
for patient_id, nct_ids in retrived_ncts.items():
    # Find the patient information
    patient_text = patient_dict.get(patient_id, "")

    # Prepare the entry for this patient
    patient_entry = {
        "patient_id": patient_id,
        "patient": patient_text,
        "0": [],
        "1": [],
        "2": []
    }

    # Iterate over the NCT IDs
    for i, nct_id in enumerate(nct_ids):
        trial = trial_info.get(nct_id, {})
        if trial:
            # Prepare the trial information
            trial_entry = {
                "brief_title": trial.get("brief_title", ""),
                "phase": trial.get("phase", ""),
                "drugs": trial.get("drugs", "[]"),
                "drugs_list": trial.get("drugs_list", []),
                "diseases": trial.get("diseases", "[]"),
                "diseases_list": trial.get("diseases_list", []),
                "enrollment": trial.get("enrollment", ""),
                "inclusion_criteria": trial.get("inclusion_criteria", ""),
                "exclusion_criteria": trial.get("exclusion_criteria", ""),
                "brief_summary": trial.get("brief_summary", ""),
                "NCTID": nct_id
            }

            # Append to the appropriate list
            patient_entry["2"].append(trial_entry)

    # Add the patient entry to the output list
    output.append(patient_entry)

# Save the output to a JSON file
with open('../../GS_sample/dataset/GS_data/pre_matching_retrived_process_file.json', 'w') as f:
    json.dump(output, f, indent=4)

print("JSON file 'pre_matching_process_file.json' has been created.")

# %%
