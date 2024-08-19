#%%
import numpy as np
import pandas as pd
import sqlite3

# Step 1: Load the .npy file
data = np.load('../GS_sample/trialgpt_retrieval/experiments_extras/sigir_embeds.npy')

# Step 2: Convert the numpy array to a pandas DataFrame
df = pd.DataFrame(data)

# Step 3: Connect to SQLite database (or create one)
conn = sqlite3.connect('your_database.sqlite')

# Step 4: Save the DataFrame to the SQLite database
df.to_sql('your_table_name', conn, if_exists='replace', index=False)

# Step 5: Close the database connection
conn.close()
#%%