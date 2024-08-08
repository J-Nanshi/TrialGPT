__author__ = "qiao"

"""
Using GPT to aggregate the scores by itself.
"""
#%%
from beir.datasets.data_loader import GenericDataLoader
import json
from nltk.tokenize import sent_tokenize
import os
import sys
import time
import logging
from TrialGPT_1 import trialgpt_aggregation

#%%
# if __name__ == "__main__":
# corpus = sys.argv[1] 
corpus = "sigir"
# model = sys.argv[2]
model = "gpt-4-turbo"
# the path of the matching results
# matching_results_path = sys.argv[3]
matching_results_path = "../results/matching_results_sigir_gpt-4-turbo_samplerun.json"
results = json.load(open(matching_results_path))
results
#%%
# loading the trial2info dict
trial2info = json.load(open("../dataset/trial_info.json"))
trial2info
#%%
# loading the patient info
_, queries, _ = GenericDataLoader(data_folder=f"../dataset/{corpus}/").load(split="test")
#%%
queries_1 = {'sigir-20141': 'A 58-year-old African-American woman presents to the ER with episodic pressing/burning anterior chest pain that began two days earlier for the first time in her life. The pain started while she was walking, radiates to the back, and is accompanied by nausea, diaphoresis and mild dyspnea, but is not increased on inspiration. The latest episode of pain ended half an hour prior to her arrival. She is known to have hypertension and obesity. She denies smoking, diabetes, hypercholesterolemia, or a family history of heart disease. She currently takes no medications. Physical examination is normal. The EKG shows nonspecific changes.'}
#%%
# output file path
output_path = f"../results/aggregation_results_{corpus}_{model}.json"
logging.basicConfig(filename='sample_log_for_ranking_agrregation.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# if os.path.exists(output_path):
# 	output = json.load(open(output_path))
# else:
output = {}


# patient-level
for patient_id, info in results.items():
	# get the patient note
	patient = queries_1[patient_id]
	sents = sent_tokenize(patient)
	sents.append("The patient will provide informed consent, and will comply with the trial protocol without any practical issues.")
	sents = [f"{idx}. {sent}" for idx, sent in enumerate(sents)]
	patient = "\n".join(sents)
	logging.info(f"patient infor:{patient}")

	if patient_id not in output:
		output[patient_id] = {}
	
	# label-level, 3 label / patient
	for label, trials in info.items():
			
		# trial-level
		for trial_id, trial_results in trials.items():
			# already cached results
			if trial_id in output[patient_id]:
				continue

			if type(trial_results) is not dict:
				output[patient_id][trial_id] = "matching result error"

				with open(output_path, "w") as f:
					json.dump(output, f, indent=4)

				continue

			# specific trial information
			trial_info = trial2info[trial_id]	

			try:
				result = trialgpt_aggregation(patient, trial_results, trial_info, model)
				output[patient_id][trial_id] = result[0]
				logging.info(f"system promt:{result[1]}, user_prompt: {result[2]}")

				with open(output_path, "w") as f:
					json.dump(output, f, indent=4)

			except:
				continue
	# 		break
	# 	break
	# break
# %%
