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
from tqdm import tqdm
import logging
import time
import tiktoken
from JN_TrialGPT import trialgpt_aggregation

#%%
# if __name__ == "__main__":
# corpus = sys.argv[1] 
corpus = "GS_data"
# model = sys.argv[2]
model = "gpt-4-turbo"
# the path of the matching results
# matching_results_path = sys.argv[3]
matching_results_path = "../../GS_sample/dataset/GS_data/token_counts/Results/P_NCT02448420_matching_results_for_5_patients_top50_trials_retrived.json"
results = json.load(open(matching_results_path))
results
#%%
# loading the trial2info dict
trial2info = json.load(open("../dataset/trial_info.json"))
trial2info
# %%
# loading the patient info
# _, queries, _ = GenericDataLoader(data_folder=f"../dataset/{corpus}/").load(split="test")
# queries
#%%
#-------IMPORTANT--------The quries/patirnt input data for matching should be given in below format
queries_1 = {"P_NCT02448420": "Emma, a 56-year-old postmenopausal woman, was diagnosed with HER2-positive, hormone receptor-positive metastatic breast cancer. She had previously received three lines of systemic treatment for her metastatic disease, including trastuzumab and chemotherapy. Sarah's most recent scans showed progression of her liver metastases, and her oncologist was considering new treatment options. She had an ECOG performance status of 1 and adequate organ function, including a baseline left ventricular ejection fraction of 55%. Sarah's tumor tissue from a recent liver biopsy was available for biomarker analysis. She had no history of other malignancies in the past five years and had not received any investigational drugs in the last two weeks. Sarah had experienced some cardiotoxicity during her previous treatments but had recovered, with no current signs of congestive heart failure or uncontrolled hypertension. She had no active infections or other severe uncontrolled medical conditions. Sarah was interested in exploring clinical trial options that could potentially offer new treatment combinations for her HER2-positive, hormone receptor-positive metastatic breast cancer."}
#%%
# output file path
output_path = f"../../GS_sample/dataset/GS_data/token_counts/Results/P_NCT02448420_LLM_aggregation_results.json"
logging.basicConfig(filename='../../GS_sample/dataset/GS_data/token_counts/Results/P_NCT02448420_ranking_LLM_agrregation.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

#%%
def num_tokens_from_string(string: str):
    encoding = tiktoken.get_encoding("cl100k_base")
    num_tokens = len(encoding.encode(string))
    return num_tokens
#%%
# if os.path.exists(output_path):
# 	output = json.load(open(output_path))
# else:
output = {}

matching_metadata_list = []
final_input_tokens = []
final_output_tokens = []
final_start_time = time.time() 
# patient-level
for patient_id, info in tqdm(results.items()):
	# print(queries_1[patient_id])
	begin = time.time() 
	# get the patient note
	patient = queries_1[patient_id]
	sents = sent_tokenize(patient)
	sents.append("The patient will provide informed consent, and will comply with the trial protocol without any practical issues.")
	sents = [f"{idx}. {sent}" for idx, sent in enumerate(sents)]
	patient = "\n".join(sents)
	logging.info(f"patient infor:{patient}")

	if patient_id not in output:
		output[patient_id] = {}
	
	input_token_list = []
	output_token_list = []
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
				results = trialgpt_aggregation(patient, trial_results, trial_info, model)
				output[patient_id][trial_id] = results[0]
				input_tokens = num_tokens_from_string(f"{results[1]+results[2]}")
				output_tokens = num_tokens_from_string(f"{results[0]}")
				input_token_list.append(input_tokens)
				output_token_list.append(output_tokens)
				final_input_tokens.append(input_tokens) 
				final_output_tokens.append(output_tokens)
				matching_metadata_dict = {"NCTID:": trial_id, "system_prompt" :results[1],"user_prompt": results[2], "input_tokens": input_tokens, "output_tokens": output_tokens}
				matching_metadata_list.append(matching_metadata_dict)
				logging.info(f"for {trial_id},\n system promt:{results[1]},\n user_prompt: {results[2]},\n input_tokens: {input_tokens},\n output_tokens: {output_tokens}")

				with open(output_path, "w") as f:
					json.dump(output, f, indent=4)

			except:
				continue
	# 		break
	end = time.time() 
	logging.info(f"For {patient_id}\n Total runtime of the program is {(end - begin)/60} minitues\n Total input tokens are: {sum(input_token_list)}\n Total output tokens are: {sum(output_token_list)}")
	# 	break
final_end_time = time.time() 
logging.info(f"Completing the matching process the stats are : Total runtime of the program is {(final_end_time - final_start_time)/60} minitues\n Total input tokens are: {sum(final_input_tokens)}\n Total output tokens are: {sum(final_output_tokens)}")
# %%
