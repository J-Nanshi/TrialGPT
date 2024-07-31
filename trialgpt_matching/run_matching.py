# __author__ = "qiao"

"""
Running the TrialGPT matching for three cohorts (sigir, TREC 2021, TREC 2022).
"""
#%%
import json
from nltk.tokenize import sent_tokenize
import os
import sys
from tqdm import tqdm
from TrialGPT import trialgpt_matching 

# corpus = sys.argv[1]
corpus = "sigir"
# model = sys.argv[2] 
model = "gpt-4-turbo" 

dataset = json.load(open(f"../dataset/{corpus}/retrieved_trials.json"))

output_path = f"../results/matching_results_{corpus}_{model}.json" 
#%%
# Dict{Str(patient_id): Dict{Str(label): Dict{Str(trial_id): Str(output)}}}
if os.path.exists(output_path):
	output = json.load(open(output_path))
else:
	output = {}

for instance in tqdm(dataset):
	# Dict{'patient': Str(patient), '0': Str(NCTID), ...}
	patient_id = instance["patient_id"]
	patient = instance["patient"]
	sents = sent_tokenize(patient)
	sents.append("The patient will provide informed consent, and will comply with the trial protocol without any practical issues.")
	sents = [f"{idx}. {sent}" for idx, sent in enumerate(sents)]
	patient = "\n".join(sents)

	# initialize the patient id in the output 
	if patient_id not in output:
		output[patient_id] = {"0": {}, "1": {}, "2": {}}
	
	for label in ["2", "1", "0"]:
		if label not in instance: continue

		for trial in instance[label]: 
			trial_id = trial["NCTID"]
			# already calculated and cached
			if trial_id in output[patient_id][label]:
				continue
			
			# in case anything goes wrong (e.g., API calling errors)
			try:
				# results, a, b = trialgpt_matching(trial, patient, model)
				results = trialgpt_matching(trial, patient, model)
				output[patient_id][label][trial_id] = results
				# with open("../results/prompts_sample.txt", "a") as f:
				# 	for item in sample:
				# 		f.write(item + "\n")

				with open(output_path, "w") as f:
					json.dump(output, f, indent=4)

			except Exception as e:
				print(e)
				continue
			# break
		# break
	# break