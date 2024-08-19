#%%
import matplotlib.pyplot as plt
import json
import pandas as pd
import os
import re
import seaborn as sns
#%%
faiss_top100 = pd.read_csv("timings/Time_taken_for_30_patient_to_get_Top100_trials.csv")
faiss_top100 = faiss_top100.drop("Patients", axis=1)
qdrant_top100 = pd.read_csv("timings/Time_taken_for_qdrant_30_patient_to_get_Top100_trials.csv")
qdrant_top100 = qdrant_top100.drop("Patients", axis=1)
# %%
final_df =  pd.concat([faiss_top100, qdrant_top100], axis=1,ignore_index=True)
final_df.columns = ["Top100_Faiss", "Top100_Qdrant"]
plt.figure(figsize=(10, 6))
# Create a box plot for all the TopK columns
sns.boxplot(data=final_df)

# Set plot title and labels
plt.title("Box Plot of Retrieval times of all 30 paitents having background as 990 trials for Top100 Ranks")
plt.xlabel("Vector Stores")
plt.ylabel("Time (s)")
plt.savefig("timings/Plots_and_dataframe/Comparision_Retrieval_times_of_all30_paitents_990trial_Top100.png")
# %%
