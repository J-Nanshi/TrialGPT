#%%
import json
import requests
from tqdm import tqdm
#%%
input_json = json.load(open(r"GS_corpus_with_gt_pan_cancer_cases_trials.json"))
input_json
# %%
result = []
for nct_id, trial_info in tqdm(input_json.items()):
    converted_dict = {
        "brief_title": trial_info["brief_title"],
        "phase": trial_info["phase"],
        "drugs": trial_info["drugs"],
        "drugs_list": trial_info["drugs_list"],
        "diseases": trial_info["diseases"],
        "diseases_list": trial_info["diseases_list"],
        "enrollment": trial_info["enrollment"],
        "inclusion_criteria": trial_info["inclusion_criteria"],
        "exclusion_criteria": trial_info["exclusion_criteria"],
        "brief_summary": trial_info["brief_summary"],
        "NCT_ID": nct_id
    }
    result.append(converted_dict)
#%%
result
#%%
#addding exclusion criteria string for uniformity.
for item in result:
    if item.get('exclusion_criteria'):  # Check if exclusion_criteria is not empty
        if not item['exclusion_criteria'].startswith('exclusion criteria:'):
            item['exclusion_criteria'] = 'exclusion criteria: ' + item['exclusion_criteria'].lstrip(' :')

#%%
result
# %%
with open(r"GS_corpus_with_gt_pan_cancer_cases_trials_tobe_used_in_GS_algo.json", "w") as f:
			json.dump(result, f, indent=4)
# %%
