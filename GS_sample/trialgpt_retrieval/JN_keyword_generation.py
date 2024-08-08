# __author__ = "qiao"

"""
generate the search keywords for each patient
"""
#%%
# !pip install openai
#%%
import openai
import json
import os
from openai import OpenAI
import sys
#%%
from configparser import ConfigParser
config = ConfigParser() 
config.read('../secrets.ini')
# %%
os.environ["OPENAI_API_KEY"] = config['OpenAI.Science-vNext-Internal']['api_key']
client = openai.OpenAI(
    # This is the default and can be omitted
    api_key=os.environ.get("OPENAI_API_KEY"),
)

def get_keyword_generation_messages(note):
	system = '''You are a helpful assistant and your task is to help search relevant clinical trials for a given patient description. P
	lease first summarize the main medical problems of the patient. Then generate up to 32 key conditions for searching relevant clinical trials for this patient. 
	The key condition list should be ranked by priority. Please output only a JSON dict formatted as Dict{{"summary": Str(summary), "conditions": List[Str(condition)]}}.'''

	prompt =  f"Here is the patient description: \n{note}\n\nJSON output:"

	messages = [
		{"role": "system", "content": system},
		{"role": "user", "content": prompt}
	]
	
	return messages
#%%

# the corpus: trec_2021, trec_2022, or sigir
#%%
# corpus = "sigir"
#%%
# the model index to use
model = "gpt-4-turbo"
#%%
import pandas as pd
import json

# Create the DataFrame
df = pd.read_csv("../GS_sample/synthetic_patient_cases_random_30_modified.csv")
# Convert the DataFrame to JSON Lines format
jsonl_data = df.apply(lambda row: json.dumps({"_id": row["NCT_ID"], "text": row["Synthetic_patients"]}), axis=1)

# Write the JSON Lines to a file
with open(r"..\GS_sample\dataset\GS_data\synthetic_patient_cases_random_30_modified.jsonl", "w") as f:
    for line in jsonl_data:
        f.write(line + "\n")
#%%
outputs = {}

# with open(r"D:\Job\TrialGPT\dataset\sigir\queries.jsonl", "r") as f:
with open(r"..\GS_sample\dataset\GS_data\synthetic_patient_cases_random_30_modified.jsonl", "r") as f:
	for line in f.readlines():
		entry = json.loads(line)
		print(f"the json file query {entry}")
		messages = get_keyword_generation_messages(entry["text"])

		response = client.chat.completions.create(
			model=model,
			messages=messages,
			temperature=0,
		)

		output = response.choices[0].message.content
		print(f"response from LLM,\n{output}")
		output = output.strip("`").strip("json")
		
		outputs[entry["_id"]] = json.loads(output)
		print(f'''the outputs[entry["_id"]] is\n {outputs[entry["_id"]]}''' )

		# with open(r"D:\Job\TrialGPT\results\retrieval_keywords_gpt4turbo_sigir.json", "w") as f:
		with open(r"..\GS_sample\results\retrieval_keywords_gpt4turbo_synthetic_patient_cases_random_30.json", "w") as f:
			json.dump(outputs, f, indent=4)
# %%
