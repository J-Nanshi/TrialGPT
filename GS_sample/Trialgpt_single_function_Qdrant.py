#%%
import os
import json
import torch
import numpy as np
import pandas as pd
from nltk import word_tokenize
from tqdm import tqdm
from rank_bm25 import BM25Okapi
from transformers import AutoTokenizer, AutoModel
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from openai import OpenAI
from configparser import ConfigParser
config = ConfigParser() 
config.read('../secrets.ini')
# %%
client = OpenAI(api_key=config['OpenAI.Science-vNext-Internal']['api_key'])
qdrant_client = QdrantClient(":memory:")
# qdrant_client = QdrantClient(host="localhost", port=6333, api_key=config["qdrant-cloud"]['api_key'], path="../GS_sample/collection/")
# qdrant_client = QdrantClient(api_key=config["qdrant-cloud"]['api_key'], path="../GS_sample/collection/")

#Essential to solve the nltk token lookup error one time use
# nltk.download('punkt')
#%%
'''The beolow functiion will generate upto 32 keywords of patients health details'''
def get_keyword_generation_messages(note):
	system = '''You are a helpful assistant and your task is to help search relevant clinical trials for a given patient description. P
	lease first summarize the main medical problems of the patient. Then generate up to 32 key conditions for searching relevant clinical trials for this patient. 
	The key condition list should be ranked by priority. Please output only a JSON dict formatted as Dict{"raw":str(patient_casestudy_input), "gpt-4-turbo": {"summary": Str(summary), "conditions": List[Str(condition)]}}.'''

	prompt =  f"Here is the patient description: \n{note}\n\nJSON output:"

	messages = [
		{"role": "system", "content": system},
		{"role": "user", "content": prompt}
	]
	
	return messages
#%%
'''This function either loads a preprocessed, tokenized corpus from a cached file or 
builds it from a JSONL file containing clinical trial entries. 
It applies token weighting based on the title (3), conditions(2), and text(1) of each trial entry. 
After processing, it saves the data for future use and creates a BM25 index to enable 
efficient text retrieval based on the BM25 algorithm. 
The function returns the BM25 index and the list of trial IDs.'''
def get_bm25_corpus_index(corpus):
	corpus_path = os.path.join(f"../trialgpt_retrieval/bm25_corpus_{corpus}.json") #Change storing the bm25 tokenized 

	# if already cached then load, otherwise build
	if os.path.exists(corpus_path):
		corpus_data = json.load(open(corpus_path))
		tokenized_corpus = corpus_data["tokenized_corpus"]
		corpus_nctids = corpus_data["corpus_nctids"]

	else:
		tokenized_corpus = []
		corpus_nctids = []

		with open("../GS_sample/dataset/GS_data/GS_trials_corpus.jsonl", "r") as f: #GS clinical trials
			for line in f.readlines():
				entry = json.loads(line)
				corpus_nctids.append(entry["_id"])
				
				# weighting: 3 * title, 2 * condition, 1 * text
				tokens = word_tokenize(entry["title"].lower()) * 3
				for disease in entry["metadata"]["diseases_list"]:
					tokens += word_tokenize(disease.lower()) * 2
				tokens += word_tokenize(entry["text"].lower())

				tokenized_corpus.append(tokens)

		corpus_data = {
			"tokenized_corpus": tokenized_corpus,
			"corpus_nctids": corpus_nctids,
		}

		with open(corpus_path, "w") as f:
			json.dump(corpus_data, f, indent=4)
	
	bm25 = BM25Okapi(tokenized_corpus)

	return bm25, corpus_nctids
#%%
'''This function processes a corpus of clinical trial data by either loading 
precomputed embeddings or computing new embeddings using the MedCPT-Article-Encoder. 
It tokenizes and encodes the text data, computes embeddings, and stores them for future use. 
Finally, it creates a FAISS index for efficient similarity search and returns the index along with the NCT IDs.'''	
def get_medcpt_corpus_index(corpus):
    corpus_path = f"../trialgpt_retrieval/{corpus}_embeds.npy"  # Change to where we want to store the data
    nctids_path = f"../trialgpt_retrieval/{corpus}_nctids.json"  # Change to where we want to store the data

    # If already cached, then load; otherwise, build
    if os.path.exists(corpus_path):
        embeds = np.load(corpus_path)
        corpus_nctids = json.load(open(nctids_path))
    else:
        embeds = []
        corpus_nctids = []

        # model = AutoModel.from_pretrained("ncbi/MedCPT-Article-Encoder").to("cuda")
        model = AutoModel.from_pretrained("ncbi/MedCPT-Article-Encoder")
        tokenizer = AutoTokenizer.from_pretrained("ncbi/MedCPT-Article-Encoder")

        with open("../GS_sample/dataset/GS_data/GS_trials_corpus.jsonl", "r") as f:  # GS clinical trials
            print("Encoding the corpus")
            for line in tqdm(f.readlines()):
                entry = json.loads(line)
                corpus_nctids.append(entry["_id"])

                title = entry["title"]
                text = entry["text"]

                with torch.no_grad():
                    # Tokenize the articles
                    encoded = tokenizer(
                        [[title, text]],
                        truncation=True,
                        padding=True,
                        return_tensors='pt',
                        max_length=512,
                        # ).to("cuda")
                    )

                    embed = model(**encoded).last_hidden_state[:, 0, :]

                    # embeds.append(embed[0].cpu().numpy())
                    embeds.append(embed[0].numpy())

        embeds = np.array(embeds)

        np.save(corpus_path, embeds)
        with open(nctids_path, "w") as f:
            json.dump(corpus_nctids, f, indent=4)

    collection_name = "clinical_trials"
    vector_dim = embeds.shape[1]

    qdrant_client.recreate_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=vector_dim, distance=Distance.COSINE),
    )

    # Insert vectors into the Qdrant collection
    points = [
        PointStruct(id=idx, vector=vector, payload={"nctid": nctid})
        for idx, (vector, nctid) in enumerate(zip(embeds, corpus_nctids))
    ]

    qdrant_client.upsert(collection_name=collection_name, points=points)

    return collection_name, corpus_nctids

#%%
def Clinical_trial_matching(CT_DB_name, TopK_retrieval, Patient_query):
    GS_trial_info = json.load(open("../GS_sample/dataset/GS_data/GS_trials_info.json"))
    corpus = CT_DB_name  # main database setup folder
    model_keyword = "gpt-4-turbo" # model used for keyword formation of patients
    k = 100  # different k for fusion
    bm25_wt = 1 # bm25 weight 
    q_type = "gpt-4-turbo" # query type 
    medcpt_wt = 1 # medcpt weight
    N = TopK_retrieval # how many to rank, gives the topk output of each patient
    Patient_text = Patient_query
    bm25, bm25_nctids = get_bm25_corpus_index(corpus)  #BM25 keyword conversion and caching only required once to run
    medcpt, medcpt_nctids = get_medcpt_corpus_index(corpus) #MedCPT vector conversion and caching only required once to run

    #loading the query encoder for MedCPT
    model_medcpt = AutoModel.from_pretrained("ncbi/MedCPT-Query-Encoder")
    tokenizer_medcpt = AutoTokenizer.from_pretrained("ncbi/MedCPT-Query-Encoder")

    #-----------------------------------------------------------------------------------------
    #Keyword generation.
    outputs_patient_keyword = {}
    messages = get_keyword_generation_messages(Patient_text)
    response = client.chat.completions.create(
                model=model_keyword,
                messages=messages,
                temperature=0,
            )
    outputs_patient = response.choices[0].message.content
    # print(f"response before processing,\n{outputs_patient}")
    outputs_patient = outputs_patient.strip("`").strip("json") # Direct input of patient data for the retrieval process 
    # print(f"response after processing,\n{outputs_patient}")
    outputs_patient_keyword["Patient_details"] = json.loads(outputs_patient)
    #------------------------------------------------------------------------------------------
    #Retrieval of the Clinical Trials
    qid2nctids = {} #
    qid = "Patient_details"

    # get the keyword list
    if q_type in outputs_patient_keyword["Patient_details"]:
        conditions = outputs_patient_keyword[qid][q_type]["conditions"]

    if len(conditions) == 0:
        nctid2score = {}
    else:
    # a list of nctid lists for the bm25 retriever
        bm25_condition_top_nctids = []

        for condition in conditions:
            tokens = word_tokenize(condition.lower())
            top_nctids = bm25.get_top_n(tokens, bm25_nctids, n=N)
            bm25_condition_top_nctids.append(top_nctids)
            # if condition[0] =="Chest pain":

        # doing MedCPT retrieval
        with torch.no_grad():
            encoded = tokenizer_medcpt(
                conditions, 
                truncation=True, 
                padding=True, 
                return_tensors='pt', 
                max_length=256,
            # ).to("cuda")
            )

            # encode the queries (use the [CLS] last hidden states as the representations)
            # embeds = model(**encoded).last_hidden_state[:, 0, :].cpu().numpy()
            embeds = model_medcpt(**encoded).last_hidden_state[:, 0, :].numpy()

        collection_name = "clinical_trials"
        medcpt_condition_top_nctids = []       # here they are retrieving the NCT_IDS for each index present in ind_list map corresponding to index of medcpt_ncts
        for embed in embeds:
            search_results = qdrant_client.search(
                collection_name= collection_name,
                query_vector=embed,
                limit=N
            )
            top_nctids = [result.payload["nctid"] for result in search_results]
            medcpt_condition_top_nctids.append(top_nctids)

        nctid2score = {}
        '''first they calculate the scores for each id with iterations for bm25 and
        they are appending the scores to the nctid2score. if the nctid is already existing and has a scrore
        the score of the id in that iteration is added to the previous score of x iteration.
        thus the same haapens for medcpt as if the nctid doesnt have a scrore it is calculated and appended 
        to the list. However if present then the score is combined with previous iterations.
        Note that - not only the scores combine id repeated in individual list but also combines if 
        they occur in both the as well
        Eg - bm25_list [ "NCT1"=0.5, "NCT2"=0.1, "NCT3"=0.6], medcpt_list ["NCT1"=0.5, "NCT4"=0.4, NCT2"=0.7]
        nctid2score = NCT1 = 0.5+0.5 = 1, NCT2 = 0.1+0.7 = 0.8, NCT3 = 0.6, NCT4 = 0.4'''

        for condition_idx, (bm25_top_nctids, medcpt_top_nctids) in enumerate(zip(bm25_condition_top_nctids, medcpt_condition_top_nctids)):


            if bm25_wt > 0:
                for rank, nctid in enumerate(bm25_top_nctids):
                    if nctid not in nctid2score:
                        nctid2score[nctid] = 0
                    
                    nctid2score[nctid] += (1 / (rank + k)) * (1 / (condition_idx + 1))

            if medcpt_wt > 0:
                for rank, nctid in enumerate(medcpt_top_nctids):
                    if nctid not in nctid2score:
                        nctid2score[nctid] = 0
                    
                    nctid2score[nctid] += (1 / (rank + k)) * (1 / (condition_idx + 1))


    nctid2score = sorted(nctid2score.items(), key=lambda x: -x[1])
    top_nctids = [nctid for nctid, _ in nctid2score[:N]]
    qid2nctids[qid] = top_nctids

    #--------------------------------------------------------------------------------------------------------

    # Printing first 5 NCT IDs in the retrieved list for the corresponding patient
    for nct_id in qid2nctids["Patient_details"][:3]:
        if nct_id in GS_trial_info:
            print("The Relevant Clinical trials for the given patient are:\n")
            print(f"Clinical Trial ID: {nct_id}")
            print(f"Brief Title: {GS_trial_info[nct_id]['brief_title']}")
            print(f"Phase: {GS_trial_info[nct_id]['phase']}")
            print(f"Inclusion Criteria: {GS_trial_info[nct_id]['inclusion_criteria']}")
            print(f"Exclusion Criteria: {GS_trial_info[nct_id]['exclusion_criteria']}")
            print("----------------------------------------------------")

# %%
corpus = "GS_data"  # main database setup folder
N = 100 # how many to rank, gives the topk output of each patient
Patient_text = "Emily, a 45-year-old woman, was diagnosed with advanced triple-negative breast cancer two years ago. She had undergone genetic testing, which revealed a BRCA1 mutation. Sarah had initially responded well to first-line chemotherapy, but her disease had recently progressed. Her oncologist had been exploring alternative treatment options when they came across this clinical trial. Sarah's cancer was measurable by RECIST 1.1 criteria, with a lesion in her liver measuring 3 cm. She had an ECOG performance status of 1 and had not participated in any other clinical trials. Sarah's most recent scans showed no evidence of brain metastases, and she had no history of autoimmune diseases or other malignancies. She had completed her last chemotherapy treatment five weeks ago and had recovered from its side effects. Sarah was postmenopausal due to chemotherapy-induced menopause and had no plans for pregnancy. Her organ function tests were within normal limits, and she had no history of HIV, Hepatitis B, or Hepatitis C. Sarah had an available formalin-fixed paraffin-embedded (FFPE) tumor tissue sample from a previous biopsy. She was not taking any medications that could interfere with the study drugs, such as strong CYP3A inhibitors or inducers. Sarah was interested in learning more about the combination of pembrolizumab and olaparib as a potential treatment option for her advanced BRCA-mutated breast cancer."
Clinical_trial_matching(corpus,N,Patient_text)
# %%
