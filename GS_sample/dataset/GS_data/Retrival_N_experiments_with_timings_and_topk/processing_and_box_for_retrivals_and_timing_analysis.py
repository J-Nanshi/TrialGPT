#%%
import matplotlib.pyplot as plt
import json

import pandas as pd
import os
import re
import seaborn as sns
#%%
directory = r"..\Retrival_N_experiments_with_timings_and_topk\retrievals"
retrieval_list = os.listdir(directory)
for i in retrieval_list:
    data = json.load(open(directory + "\\"+ i))

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
    match= re.search(r"N(\d+)", i).group(1)
    df = pd.DataFrame({"Patients": x_labels, f"Ranks_Top_{match}": [i+1 for i in y_values]})
    # df.to_csv(f"retrievals_df/Top{match}_999trials_all30_patients_df.csv", index=None)
#%%
#box plot of all topk
topk_directory = r"..\Retrival_N_experiments_with_timings_and_topk\retrievals_df"
topk_list = os.listdir(topk_directory)
topK_list = topk_list.pop(0)
consolidated_df = pd.DataFrame()
Top10 = pd.read_csv("retrievals_df\Top10_999trials_all30_patients_df.csv")
patient_list = list(Top10["Patients"])

new_columns_name = []
for i in topk_list:
    file_path = os.path.join(topk_directory, i)
    new_column_name = i.split("_")[0] 
    df = pd.read_csv(file_path, encoding='ISO-8859-1')
    df = df.drop("Patients", axis=1)
    new_columns_name.append(new_column_name)
    consolidated_df = pd.concat([consolidated_df, df], axis=1,ignore_index=True)


consolidated_df.index = patient_list
consolidated_df.columns = new_columns_name
first_column = consolidated_df.pop(consolidated_df.columns[0])  # Pop the first column
consolidated_df['Top100'] = first_column 
consolidated_df
consolidated_df.to_csv("retrievals_df/plots_and_dataframe/Consolidated_ranks_all30_patients_999trials_df.csv")
#%%
plt.figure(figsize=(10, 6))
consolidated_df = pd.read_csv("retrievals_df/plots_and_dataframe/Consolidated_ranks_all30_patients_999trials_df.csv", index_col=0)
consolidated_df
#%%
# Create a box plot for all the TopK columns
sns.boxplot(data=consolidated_df)

# Set plot title and labels
plt.title("Box Plot of Ranks for TopK Values of all 30 paitents having background as 990 trials")
plt.xlabel("TopK")
plt.ylabel("Rank, NOTE- Rank 0 = NOT MATCHED")
plt.savefig("retrievals_df/plots_and_dataframe/Consolidated_ranks_box_plot_from_top10_top100_for_all30_patients_999trials_df.png")
#%%
counts = (consolidated_df.iloc[:,:] > 0).sum()

# Calculate the percentage of non-zero values
percentages = (counts / len(consolidated_df)) * 100

# Plotting
plt.figure(figsize=(18, 15))
bars = plt.bar(counts.index, counts)

# Add the count and percentage as text on top of the bars
for bar, count, percentage in zip(bars, counts, percentages):
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.2, f'{count}\n({percentage:.1f}%)', ha='center', va='bottom')

# Adding labels and title
plt.xlabel('TopK', fontsize=14)
plt.ylabel('Count of trials matched', fontsize=14)
plt.title('Bar plot of sum of matched trials and their fraction for each topK, total patients=30', fontsize=16)
plt.savefig("retrievals_df/plots_and_dataframe/Sum_of_matched_trials_and_their_fraction_for_each_topk_for_all30P_999_trials.pdf")
plt.savefig("retrievals_df/plots_and_dataframe/Sum_of_matched_trials_and_their_fraction_for_each_topk_for_all30P_999_trials.png")
plt.xticks(fontsize=12)  # Adjusting font size for x-ticks
plt.yticks(fontsize=12)
plt.show()
#%%
timings_directory = "../Retrival_N_experiments_with_timings_and_topk/timings"
timings_list = os.listdir(timings_directory)
timings_list.pop(0)
timings_list.pop(-1)
print(timings_list)
consolidated_time_df = pd.DataFrame()
top10_df = pd.read_csv(r"timings\Time_taken_for_30_patient_to_get_Top10_trials.csv")
patient_list_time = list(top10_df["Patients"])

new_columns_name = []
for i in timings_list:
    file_path = os.path.join(timings_directory, i)
    new_column_name = i.split("_")[7] 
    df = pd.read_csv(file_path, encoding='ISO-8859-1')
    df = df.drop("Patients", axis=1)
    new_columns_name.append(new_column_name)
    consolidated_time_df = pd.concat([consolidated_time_df, df], axis=1,ignore_index=True)


consolidated_time_df.index = patient_list_time
# print(consolidated_time_df)
consolidated_time_df.columns = new_columns_name
first_column = consolidated_time_df.pop(consolidated_time_df.columns[0])  # Pop the first column
consolidated_time_df['Top100'] = first_column 
print(consolidated_time_df)
consolidated_time_df.to_csv("timings/Plots_and_dataframe/Consolidated_time_topK_all30_patients_999trials_df.csv")
# %%
plt.figure(figsize=(10, 6))
consolidated_time_df = pd.read_csv("timings/Plots_and_dataframe/Consolidated_time_topK_all30_patients_999trials_df.csv")
# Create a box plot for all the TopK columns
sns.boxplot(data=consolidated_time_df)

# Set plot title and labels
plt.title("Box Plot of Time for TopK Values of all 30 paitents having backgrpund as 999 trials")
plt.xlabel("TopK")
plt.ylabel("Time (s)")
plt.savefig("timings/Plots_and_dataframe/Consolidated_times_box_plot_from_top10_top100_for_all30_patients_999trials_df.png")
