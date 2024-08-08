#%%
import json
#%%

patients_5 = ["P_NCT02583828", "P_NCT03500380","P_NCT02716116", "P_NCT02448420", "P_NCT02615457"]

#retrieving the keyword output for the given 5 patient ncts (json)
keyword_patients_5 = json.load(open(f"../../GS_sample/results/retrieval_keywords_gpt4turbo_synthetic_patient_cases_random_30.json"))
extracted_entries = {key: value for key, value in keyword_patients_5.items() if key in patients_5}

with open("../../GS_sample/dataset/GS_data/token_counts/retrieval_keywords_for_5_patient.json", "w") as f:
    json.dump(extracted_entries, f, indent=4)
# %%
#retriveing the patient raw information (jsonl)
synthetic_patient_data = []
with open("../../GS_sample/dataset/GS_data/synthetic_patient_cases_random_30_modified.jsonl", 'r') as f:
    for line in f:
        synthetic_patient_data.append(json.loads(line))
synthetic_patient_data

patient_raw = []
for entry in synthetic_patient_data:
     if entry["_id"] in patients_5:
         patient_raw.append(entry)
patient_raw

with open("../../GS_sample/dataset/GS_data/token_counts/5_selected_syntetic_patient_for_token_count.jsonl", 'w') as f:
    for entry in patient_raw:
        f.write(json.dumps(entry) + '\n')

# %%
#reteriving the id2queries format for the 5 patients (json)
id2queries_patients_5 = json.load(open(f"../../GS_sample/dataset/GS_data/GS_id2quries_all_30_synthetic_patients.json"))
extracted_entries = {key: value for key, value in id2queries_patients_5.items() if key in patients_5}

with open("../../GS_sample/dataset/GS_data/token_counts/id2queries_for_5_patient.json", "w") as f:
    json.dump(extracted_entries, f, indent=4)
#%%
#converting the id2queries of 5 patients to the pre matching requierd input.
with open('../../GS_sample/dataset/GS_data/token_counts/Results/qid2nctids_retrieval_results_for_5_patients_GS_data_k50_bm25wt1_medcptwt1_N50.json') as f:
    retrived_ncts = json.load(f)
retrived_ncts

#loading the synthetic patient raw detials
with open('../../GS_sample/dataset/GS_data/token_counts/5_selected_syntetic_patient_for_token_count.jsonl') as f:
    patient_queries = [json.loads(line) for line in f]
patient_queries

# loading TrialGPT clinical trial info
with open('../../GS_sample/dataset/trial_info.json') as f:
    trial_info = json.load(f)
trial_info

# Process patient queries into a dictionary
patient_dict = {item['_id']: item['text'] for item in patient_queries}
patient_dict
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
with open('../../GS_sample/dataset/GS_data/token_counts/pre_matching_input_5_patient_file.json', 'w') as f:
    json.dump(output, f, indent=4)
# %%
