#%%
# !pip install matplotlib
#%%
import matplotlib.pyplot as plt
import json
import pandas as pd
#%%
# Given dictionary
data = json.load(open(f"../../GS_sample/dataset/GS_data/GS_100_trials/qid2nctids_retrieval_results_for_100_active_trials_GS_data_k50_bm25wt1_medcptwt1_N100.json"))
data
#%%
# Prepare data for plotting
x_labels = []
y_values = []

for p_nctid, nctid_list in data.items():
    nctid_key = p_nctid.split('_')[1]  # Extract the NCT ID from the P_NCTID
    if nctid_key in nctid_list:
        index_position = nctid_list.index(nctid_key)
    else:
        index_position = -1  # If the NCTID is not found in the list

    x_labels.append(p_nctid)
    y_values.append(index_position)

# Plotting the scatter plot
plt.figure(figsize=(12, 8))
plt.scatter(x_labels, y_values, color='skyblue')

# Add horizontal line at y=30
plt.axhline(y=30, color='red', linestyle='--', label='y=30')

plt.xlabel('Synthetic Patients')
plt.ylabel('Rank of NCT amongst top 100 retrieved trials')
plt.title('Retrieval Ranking of NCTID')
plt.xticks(rotation=90)
plt.ylim(-1, 100)

# Add value annotations near the points
for i in range(len(y_values)):
    plt.text(i, y_values[i] + 0.5, str(y_values[i]), ha='center')

plt.legend()
plt.tight_layout()
plt.savefig("../../GS_sample/dataset/GS_data/GS_100_trials/Top100plot_100active_trials_all30_pateints.png")
plt.show()

# %%
df = pd.DataFrame({"Patients": x_labels, "Ranks_TrialGPT": [i+1 for i in y_values]})
df.to_csv("../../GS_sample/dataset/GS_data/GS_100_trials/top100_100active_trials_all30_patients_df.csv", index=None)

# %%
