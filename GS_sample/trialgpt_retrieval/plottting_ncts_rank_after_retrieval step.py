#%%
# !pip install matplotlib
#%%
import matplotlib.pyplot as plt
import json
#%%
# Given dictionary
data = json.load(open(f"../../GS_sample/results/qid2nctids_all_30_patients_results_GS_data_k50_bm25wt1_medcptwt1_N700.json"))
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
plt.axhline(y=50, color='red', linestyle='--', label='y=50')

plt.xlabel('Synthetic Patients')
plt.ylabel('Rank of NCT amongst top 700 retrieved trials')
plt.title('Retrieval Ranking of NCTID')
plt.xticks(rotation=90)
plt.ylim(-1, 600)

# Add value annotations near the points
for i in range(len(y_values)):
    plt.text(i, y_values[i] + 0.5, str(y_values[i]), ha='center')

plt.legend()
plt.tight_layout()
plt.savefig("../../GS_sample/results/Top_700_plot_for_all_30_patients_retrieved_step.png")
plt.show()

# %%
