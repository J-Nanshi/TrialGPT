# __author__ = "qiao"

"""
Conduct the first stage retrieval by the hybrid retriever 
"""
#%%
from beir.datasets.data_loader import GenericDataLoader
import faiss
import json
import nltk
from nltk import word_tokenize
import numpy as np
import os
from rank_bm25 import BM25Okapi
import sys
import tqdm
import torch
from transformers import AutoTokenizer, AutoModel

#%%
#Essential to solve the nltk token lookup error one time use
nltk.download('punkt')
#%%
#%%
'''This function either loads a preprocessed, tokenized corpus from a cached file or 
builds it from a JSONL file containing clinical trial entries. 
It applies token weighting based on the title (3), conditions(2), and text(1) of each trial entry. 
After processing, it saves the data for future use and creates a BM25 index to enable 
efficient text retrieval based on the BM25 algorithm. 
The function returns the BM25 index and the list of trial IDs.'''
def get_bm25_corpus_index(corpus):
	corpus_path = os.path.join(f"../trialgpt_retrieval/bm25_corpus_{corpus}_active100.json")

	# if already cached then load, otherwise build
	if os.path.exists(corpus_path):
		corpus_data = json.load(open(corpus_path))
		tokenized_corpus = corpus_data["tokenized_corpus"]
		corpus_nctids = corpus_data["corpus_nctids"]

	else:
		tokenized_corpus = []
		corpus_nctids = []

		with open(r"D:\Job\TrialGPT\GS_sample\dataset\GS_data\GS_100_trials\100_trials_corpus.jsonl", "r") as f:
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
	corpus_path = f"../trialgpt_retrieval/{corpus}_embeds_active100.npy" 
	nctids_path = f"../trialgpt_retrieval/{corpus}_nctids_active100.json"

	# if already cached then load, otherwise build
	if os.path.exists(corpus_path):
		embeds = np.load(corpus_path)
		corpus_nctids = json.load(open(nctids_path)) 

	else:
		embeds = []
		corpus_nctids = []

		# model = AutoModel.from_pretrained("ncbi/MedCPT-Article-Encoder").to("cuda")
		model = AutoModel.from_pretrained("ncbi/MedCPT-Article-Encoder")
		tokenizer = AutoTokenizer.from_pretrained("ncbi/MedCPT-Article-Encoder")

		with open(r"D:\Job\TrialGPT\GS_sample\dataset\GS_data\GS_100_trials\100_trials_corpus.jsonl", "r") as f:
			print("Encoding the corpus")
			for line in tqdm.tqdm(f.readlines()):
				entry = json.loads(line)
				corpus_nctids.append(entry["_id"])

				title = entry["title"]
				text = entry["text"]

				with torch.no_grad():
					# tokenize the articles
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

	index = faiss.IndexFlatIP(768)
	index.add(embeds)
	
	return index, corpus_nctids
	
#%%
# different corpora, "trec_2021", "trec_2022", "sigir"
# corpus = sys.argv[1]
corpus = "GS_data"
# query type
# q_type = sys.argv[2]
q_type = "gpt-4-turbo"
# different k for fusion
# k = int(sys.argv[3])
k = 50

# bm25 weight 
# bm25_wt = int(sys.argv[4])
bm25_wt = 1

# medcpt weight
# medcpt_wt = int(sys.argv[5])
medcpt_wt = 1

# how many to rank
N = 100

#%%
# # loading the qrels
# _, _, qrels = GenericDataLoader(data_folder=f"D:\Job\TrialGPT\dataset\sigir").load(split="test")
# qrels
#%%
# loading all types of queries
id2queries = json.load(open(f"D:\Job\TrialGPT\GS_sample\dataset\GS_data\GS_id2quries_all_30_synthetic_patients.json"))
id2queries
#%%
# loading the indices
bm25, bm25_nctids = get_bm25_corpus_index(corpus) #bm25 the indices and bm25_nctids contains the nctids of clinical trial
#%%
#%%
medcpt, medcpt_nctids = get_medcpt_corpus_index(corpus)
#%%
#%%
# loading the query encoder for MedCPT
# model = AutoModel.from_pretrained("ncbi/MedCPT-Query-Encoder").to("cuda")
model = AutoModel.from_pretrained("ncbi/MedCPT-Query-Encoder")
tokenizer = AutoTokenizer.from_pretrained("ncbi/MedCPT-Query-Encoder")
#%%
# then conduct the searches, saving top 1k
output_path = f"../../GS_sample/dataset/GS_data/GS_100_trials/qid2nctids_retrieval_results_for_100_active_trials_{corpus}_k{k}_bm25wt{bm25_wt}_medcptwt{medcpt_wt}_N{N}.json"
output_path
#%%
qid2nctids = {}
recalls = []
with open(r"D:\Job\TrialGPT\GS_sample\trialgpt_retrieval\synthetic_patient_cases_random_30_modified.jsonl", "r") as f:
	for line in tqdm.tqdm(f.readlines()):
		entry = json.loads(line)
		query = entry["text"]
		qid = entry["_id"]

		# if qid not in qrels:
			# continue

		# truth_sum = sum(qrels[qid].values())
		
		# get the keyword list
		if q_type in ["raw", "human_summary"]:
			conditions = [id2queries[qid][q_type]]
		elif "turbo" in q_type:
			conditions = id2queries[qid][q_type]["conditions"]
		elif "Clinician" in q_type:
			conditions = id2queries[qid].get(q_type, [])

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
					# break 
			# doing MedCPT retrieval
			with torch.no_grad():
				encoded = tokenizer(
					conditions, 
					truncation=True, 
					padding=True, 
					return_tensors='pt', 
					max_length=256,
				# ).to("cuda")
				)

				# encode the queries (use the [CLS] last hidden states as the representations)
				# embeds = model(**encoded).last_hidden_state[:, 0, :].cpu().numpy()
				embeds = model(**encoded).last_hidden_state[:, 0, :].numpy()

				# search the Faiss index
				scores, inds = medcpt.search(embeds, k=N) #'''getting the scores and indices of the clinial trials stored in faiss'''

			medcpt_condition_top_nctids = []

			for ind_list in inds:
				top_nctids = [medcpt_nctids[ind] for ind in ind_list] #'''here they are doing for each index present in ind_list map and retrive the the nctid corresponding to index of medcpt_ncts'''
				medcpt_condition_top_nctids.append(top_nctids)
				# break

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
						# break

				if medcpt_wt > 0:
					for rank, nctid in enumerate(medcpt_top_nctids):
						if nctid not in nctid2score:
							nctid2score[nctid] = 0
						
						nctid2score[nctid] += (1 / (rank + k)) * (1 / (condition_idx + 1))
						# break
				# break
	
		nctid2score = sorted(nctid2score.items(), key=lambda x: -x[1])
		top_nctids = [nctid for nctid, _ in nctid2score[:N]]
		qid2nctids[qid] = top_nctids

        #Caluculating the precesion of each patient not necessary currently 
		# actual_sum = sum([qrels[qid].get(nctid, 0) for nctid in top_nctids])
		# recalls.append(actual_sum / truth_sum)
		# break

with open(output_path, "w") as f:
	json.dump(qid2nctids, f, indent=4)

# %%
