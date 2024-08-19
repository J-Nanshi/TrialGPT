# __author__ = "qiao"
"""
Running the TrialGPT matching for three cohorts (sigir, TREC 2021, TREC 2022).
"""
#%%
import json
from nltk.tokenize import sent_tokenize
import os
import sys
from groq import Groq
from tqdm import tqdm
import logging
import time
import tiktoken
from JN_TrialGPT import trialgpt_matching 

# corpus = sys.argv[1]
corpus = "GS_data"
# model = sys.argv[2] 
# model = "gpt-4-turbo" 
model = "llama3-8b-8192"

dataset = json.load(open(f"../../GS_sample/dataset/GS_data/token_counts/pre_matching_input_5_patient_file.json"))
dataset[0]

output_path = f"../../GS_sample/dataset/GS_data/token_counts/Results/matching_results_for_5_patients_top50_trials_retrived_qroq.json" 
logging.basicConfig(filename='../../GS_sample/dataset/GS_data/token_counts/Results/P_NCT02448420_matching_algo_groq.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
#%%
def num_tokens_from_string(string: str):
    encoding = tiktoken.get_encoding("cl100k_base")
    num_tokens = len(encoding.encode(string))
    return num_tokens
#%%
# Dict{Str(patient_id): Dict{Str(label): Dict{Str(trial_id): Str(output)}}}
if os.path.exists(output_path):
	output = json.load(open(output_path))
else:
	output = {}

matching_metadata_list = []
final_input_tokens = []
final_output_tokens = []
final_start_time = time.time() 
for instance in tqdm(dataset):
	begin = time.time() 
	# Dict{'patient': Str(patient), '0': Str(NCTID), ...}
	patient_id = instance["patient_id"]
	patient = instance["patient"]
	sents = sent_tokenize(patient)
	sents.append("The patient will provide informed consent, and will comply with the trial protocol without any practical issues.")
	sents = [f"{idx}. {sent}" for idx, sent in enumerate(sents)]
	patient = "\n".join(sents)
	logging.info(f"patient infor:{patient}")
	# initialize the patient id in the output 
	if patient_id not in output:
		output[patient_id] = {"0": {}, "1": {}, "2": {}}
	
	input_token_list = []
	output_token_list = []
	for label in tqdm(["2", "1", "0"]):
		if label not in instance: continue

		for trial in tqdm(instance[label]): 
			trial_id = trial["NCTID"]
			# already calculated and cached
			if trial_id in output[patient_id][label]:
				continue
			
			# in case anything goes wrong (e.g., API calling errors)
			try:
				
				# results, a, b = trialgpt_matching(trial, patient, model)
				results = trialgpt_matching(trial, patient)
				output[patient_id][label][trial_id] = results[0]
				input_tokens = num_tokens_from_string(f"{results[1]+results[2]}")
				output_tokens = num_tokens_from_string(f"{results[0]}")
				input_token_list.append(input_tokens)
				output_token_list.append(output_tokens)
				final_input_tokens.append(input_tokens)
				final_output_tokens.append(output_tokens)
				matching_metadata_dict = {"NCTID:": trial_id, "system promt" :results[1],"user_prompt": results[2], "input_tokens": input_tokens, "output_tokens": output_tokens}
				matching_metadata_list.append(matching_metadata_dict)
				logging.info(f"for {trial_id},\n system promt:{results[1]},\n user_prompt: {results[2]},\n input_tokens: {input_tokens},\n output_tokens: {output_tokens}")
				
				with open(f"../GS_sample/dataset/GS_data/token_counts/Result/{patient_id}_matching_metadeta.jsonl", "w") as f:
					f.write(matching_metadata_dict + "\n")

				with open(output_path, "w") as f:
					json.dump(output, f, indent=4)

				# time.sleep(5) 	
			except Exception as e:
				logging.error(f"Error processing trial {trial_id} for patient {patient_id}: {e}")
				continue

	else:
		logging.error(f"Expected instance to be dict, but got {type(instance)}")
		# break
	# break
	end = time.time() 
	logging.info(f"For {patient_id}\n Total runtime of the program is {(end - begin)/60} minitues\n Total input tokens are: {sum(input_token_list)}\n Total output tokens are: {sum(output_token_list)}")
final_end_time = time.time() 
logging.info(f"Completing the matching process the stats are : Total runtime of the program is {(final_end_time - final_start_time)/60} minitues\n Total input tokens are: {sum(final_input_tokens)}\n Total output tokens are: {sum(final_output_tokens)}")
# %%
# %%

# corrected_json = json.load(open(f""))
# with open('corrected_file.json', 'w') as f:
#     json.dump(corrected_json, f, indent=4)

# #%%
# os.environ["GROQ_API_KEY"] = "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"
# # client = Groq(api_key="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")

# client = Groq(
#     # This is the default and can be omitted
#     api_key=os.environ.get("GROQ_API_KEY"),
# )

# chat_completion = client.chat.completions.create(
#     messages=[
#         {
#             "role": "user",
#             "content": "Explain the importance of low latency LLMs",
#         }
#     ],
#     model="llama3-8b-8192",
# )
# print(chat_completion.choices[0].message.content)
# %%
