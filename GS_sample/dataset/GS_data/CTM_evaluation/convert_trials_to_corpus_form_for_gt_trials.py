#%%
import json
#%%
# Input JSON 
input_json = json.load(open(r"GS_corpus_with_gt_pan_cancer_cases_trials.json"))
input_json
#%%
# Prepare Output JSONL Line
output = []
for key, value in input_json.items():
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
with open(r"GS_corpus_with_gt_pan_cancer_cases_trials.jsonl", 'w') as f:
    for entry in output:
        f.write(json.dumps(entry) + '\n')
# %%
output

# %%
