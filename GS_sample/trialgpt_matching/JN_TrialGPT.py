__author__ = "qiao"

"""
TrialGPT-Matching main functions.
"""
#%%
import json
from nltk.tokenize import sent_tokenize
import time
import os
from groq import Groq
# from openai import OpenAI
from configparser import ConfigParser

# config = ConfigParser() 
# config.read('../secrets.ini')
# os.environ["OPENAI_API_KEY"] = config['OpenAI.Science-vNext-Internal']['api_key']
# client = OpenAI(
#     # This is the default and can be omitted
#     api_key=os.environ.get("OPENAI_API_KEY"),
# )
os.environ["GROQ_API_KEY"] = "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"
client = Groq(
    # This is the default and can be omitted
    api_key=os.environ.get("GROQ_API_KEY")
)
#%%
def parse_criteria(criteria):
	output = ""
	criteria = criteria.split("\n\n")
	
	idx = 0
	for criterion in criteria:
		criterion = criterion.strip()

		if "inclusion criteria" in criterion.lower() or "exclusion criteria" in criterion.lower():
			continue

		if len(criterion) < 5:
			continue
	
		output += f"{idx}. {criterion}\n" 
		idx += 1
	
	return output


def print_trial(
	trial_info: dict,
	inc_exc: str,
) -> str:
	"""Given a dict of trial information, returns a string of trial."""
	
	trial = f"Title: {trial_info['brief_title']}\n"
	trial += f"Target diseases: {', '.join(trial_info['diseases_list'])}\n"
	trial += f"Interventions: {', '.join(trial_info['drugs_list'])}\n"
	trial += f"Summary: {trial_info['brief_summary']}\n"
	
	if inc_exc == "inclusion":
		trial += "Inclusion criteria:\n %s\n" % parse_criteria(trial_info['inclusion_criteria'])
	elif inc_exc == "exclusion":
		trial += "Exclusion criteria:\n %s\n" % parse_criteria(trial_info['exclusion_criteria']) 

	return trial


def get_matching_prompt(
	trial_info: dict,
	inc_exc: str,
	patient: str,
) -> str:
	"""Output the prompt."""
	prompt = f"You are a helpful assistant for clinical trial recruitment. Your task is to compare a given patient note and the {inc_exc} criteria of a clinical trial to determine the patient's eligibility at the criterion level.\n"

	if inc_exc == "inclusion":
		prompt += "The factors that allow someone to participate in a clinical study are called inclusion criteria. They are based on characteristics such as age, gender, the type and stage of a disease, previous treatment history, and other medical conditions.\n"
	
	elif inc_exc == "exclusion":
		prompt += "The factors that disqualify someone from participating are called exclusion criteria. They are based on characteristics such as age, gender, the type and stage of a disease, previous treatment history, and other medical conditions.\n"

	prompt += f"You should check the {inc_exc} criteria one-by-one, and output the following three elements for each criterion:\n"
	prompt += f"\tElement 1. For each {inc_exc} criterion, briefly generate your reasoning process: First, judge whether the criterion is not applicable (not very common), where the patient does not meet the premise of the criterion. Then, check if the patient note contains direct evidence. If so, judge whether the patient meets or does not meet the criterion. If there is no direct evidence, try to infer from existing evidence, and answer one question: If the criterion is true, is it possible that a good patient note will miss such information? If impossible, then you can assume that the criterion is not true. Otherwise, there is not enough information.\n"
	prompt += f"\tElement 2. If there is relevant information, you must generate a list of relevant sentence IDs in the patient note. If there is no relevant information, you must annotate an empty list.\n" 
	prompt += f"\tElement 3. Classify the patient eligibility for this specific {inc_exc} criterion: "
	
	if inc_exc == "inclusion":
		prompt += 'the label must be chosen from {"not applicable", "not enough information", "included", "not included"}. "not applicable" should only be used for criteria that are not applicable to the patient. "not enough information" should be used where the patient note does not contain sufficient information for making the classification. Try to use as less "not enough information" as possible because if the note does not mention a medically important fact, you can assume that the fact is not true for the patient. "included" denotes that the patient meets the inclusion criterion, while "not included" means the reverse.\n'
	elif inc_exc == "exclusion":
		prompt += 'the label must be chosen from {"not applicable", "not enough information", "excluded", "not excluded"}. "not applicable" should only be used for criteria that are not applicable to the patient. "not enough information" should be used where the patient note does not contain sufficient information for making the classification. Try to use as less "not enough information" as possible because if the note does not mention a medically important fact, you can assume that the fact is not true for the patient. "excluded" denotes that the patient meets the exclusion criterion and should be excluded in the trial, while "not excluded" means the reverse.\n'
	
	prompt += "You should output only a JSON dict exactly formatted as: dict{str(criterion_number): list[str(element_1_brief_reasoning), list[int(element_2_sentence_id)], str(element_3_eligibility_label)]}."
	
	user_prompt = f"Here is the patient note, each sentence is led by a sentence_id:\n{patient}\n\n" 
	user_prompt += f"Here is the clinical trial:\n{print_trial(trial_info, inc_exc)}\n\n"
	user_prompt += f"Plain JSON output:"

	return prompt, user_prompt


def trialgpt_matching(trial: dict, patient: str):
	results = {}

	# doing inclusions and exclusions in separate prompts
	for inc_exc in ["inclusion", "exclusion"]:
		system_prompt, user_prompt = get_matching_prompt(trial, inc_exc, patient)
		print(f"system_promt:\n {system_prompt}, \nuser_promt:\n {user_prompt}")
		messages = [
			{"role": "system", "content": system_prompt},
			{"role": "user", "content": user_prompt},
		]
		# print(message)
		response = client.chat.completions.create(
			model="llama3-8b-8192",
			messages=messages,
			temperature=0,
		)
		
		results = response.choices[0].message.content.strip()
		results = results.strip("`").strip("json")
		# time.sleep(sleep_duration)

		try:
			results[inc_exc] = json.loads(results)
		except:
			results[inc_exc] = results
		# print(system_prompt, user_prompt)

	return results, system_prompt, user_prompt
	# return [results, system_prompt, user_prompt]
#%%
# model = "gpt-4-turbo"
# trial = {'brief_title': 'A Cluster Randomized Trial to Assess the Impact of Opinion Leader Endorsed Evidence Summaries on Improving Quality of Prescribing for Patients With Chronic Cardiovascular Disease',
#  'phase': '',
#  'drugs': "['Opinion leader generated and endorsed evidence summaries']",
#  'drugs_list': ['Opinion leader generated and endorsed evidence summaries'],
#  'diseases': "['Coronar y Disease', 'Ischemic Heart Disease', 'Heart Failure']",
#  'diseases_list': ['Coronary Disease',
#   'Ischemic Heart Disease',
#   'Heart Failure'],
#  'enrollment': '160.0',
#  'inclusion_criteria': 'inclusion criteria: \n\n Patients with HF or IHD who are not currently taking the study medications of interest (ACE inhibitors/angiotensin receptor blockers for HF or statins for IHD) and whose primary care physicians are part of the study population \n\n ',
#  'exclusion_criteria': ': \n\n Patients who are unable or unwilling to give informed consent, \n\n previously taken the study medications according to dispensing records \n\n allergy or intolerance to study medications \n\n residents of long-term care facilities \n\n unable to confirm a diagnosis of either HF or IHD \n\n primary care physician has already contributed 5 patients to the study',
#  'brief_summary': 'BACKGROUND: Although much has been written about the influence of local opinion leaders on clinical practice, there have been few controlled studies of their effect, and almost none have attempted to change prescribing in the community for chronic conditions such as congestive heart failure (CHF) or ischemic heart disease (IHD). These two conditions are common and there is very good evidence about how to best prevent morbidity and mortality - and very good evidence that quality of care is, in general, suboptimal. Practice audits have demonstrated that about half of eligible CHF patients are prescribed ACE inhibitors (and fewer still reaching appropriate target doses) and less than one-third of patients with established IHD are prescribed statins (with many fewer reaching recommended cholesterol targets). It is apparent that interventions to improve quality of prescribing are urgently needed.~HYPOTHESIS: An intervention that consists of patient-specific one-page evidence summaries, generated and then endorsed by local opinion leaders, will be able to change prescribing practices of community-based primary care physicians.~DESIGN: A single centre randomized controlled trial comparing an opinion leader intervention to usual care. Based on random allocation of all physicians in one large Canadian health region, patients with CHF or IHD (not receiving ACE inhibitors or statins, respectively) recruited from community pharmacies will be allocated to intervention or usual care. The primary outcome is improvement in prescription of proven efficacious therapies for CHF (ACE inhibitors) or IHD (statins) within 6 months of the intervention.',
#  'NCTID': 'NCT00175279'}
# patient = "A 58-year-old African-American woman presents to the ER with episodic pressing/burning anterior chest pain that began two days earlier for the first time in her life.\n1. The pain started while she was walking, radiates to the back, and is accompanied by nausea, diaphoresis and mild dyspnea, but is not increased on inspiration.\n2. The latest episode of pain ended half an hour prior to her arrival.\n3. She is known to have hypertension and obesity.\n4. She denies smoking, diabetes, hypercholesterolemia, or a family history of heart disease.\n5. She currently takes no medications.\n6. Physical examination is normal.\n7. The EKG shows nonspecific changes.\n8. The patient will provide informed consent, and will comply with the trial protocol without any practical issues."
# #%%
# sample = trialgpt_matching(trial, patient, model)

# # %%
# sample
# %%
