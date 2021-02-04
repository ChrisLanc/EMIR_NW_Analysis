

'''This script provides the python implementations for most of the data preprocessing, note that some of the preprocessing steps such as the trade matching and subsampling of 
uncleared OTC transactions has been done on the database in SQL and are not shown here.
The script was origninally constructed as a Jupyter Notebook, as specific data and results cannot be shown it is provided in classical format instead.'''

# In[589]:


import pandas as pd
import os
import csv
import sys
import re
import glob
import zipfile
import io
import datetime
import time
import numpy as np
from collections import Counter
from zipfile import ZipFile
from scipy.stats import pearsonr





# In[399]:


file = r"file_path" # placeholder for the acctual filepath


# In[414]:


df = pd.read_csv(file, encoding = "utf-16", sep = "\t", dtype = object, error_bad_lines = False).fillna(np.nan)


# # Data Cleaning

# ### Drop "dead" trades with no valuation update in last 100 days

# First convert timestamps to datetime object, take the difference of reporting TS and Vauation TS and convert difference to days, then subsequently subsample based on threshold.

# In[443]:


df.loc[:,"S_TIMESTAMP_VALUATION"] = pd.to_datetime(df["S_TIMESTAMP_VALUATION"], errors = "coerce", infer_datetime_format = True)
df.loc[:,"B_TIMESTAMP_VALUATION"] = pd.to_datetime(df["B_TIMESTAMP_VALUATION"], errors = "coerce", infer_datetime_format = True)


# In[444]:


df.loc[:,"Common_TIMESTAMP_REPORTING"] = pd.to_datetime(df["Common_TIMESTAMP_REPORTING"], errors = "coerce", infer_datetime_format = True)


# In[445]:


df.loc[:,"S_val_diff"] = (df["Common_TIMESTAMP_REPORTING"][0] - df["S_TIMESTAMP_VALUATION"])
df.loc[:,"S_val_diff"] = df["S_val_diff"].apply(lambda x: x.days)


# In[446]:


df.loc[:,"B_val_diff"] = (df["Common_TIMESTAMP_REPORTING"][0] - df["B_TIMESTAMP_VALUATION"])
df.loc[:,"B_val_diff"] = df["B_val_diff"].apply(lambda x: x.days)


# #### Number of trade older than 100 days counting from latest reporting timestamp

# In[447]:


sum(df["S_val_diff"] >= 100)


# In[448]:


sum(df["B_val_diff"] >= 100)


# In[449]:


sum(((df["B_val_diff"] >= 100) | (df["S_val_diff"] >= 100)))


# In[450]:


drop_diff = df[(df["B_val_diff"] >= 100) | (df["S_val_diff"] >= 100)].index

df.drop(drop_diff, inplace = True)


# In[451]:


df.reset_index(drop = True, inplace = True)


# In[341]:


len(df)


# ### drop trades with invalid/other asset classes

# In[430]:


df.dropna(axis = 0, subset = ['Common_CLASS_ASSET'], inplace = True)
df.reset_index(drop = True, inplace = True)


# In[533]:


other_class = np.where(df['Common_CLASS_ASSET'] == 'OTHR')[0]


# In[535]:


df.drop(other_class, inplace = True)
df.reset_index(drop = True, inplace = True)


# ## Valuation of Contract

# Clean the reported valuations, plot distribution and remove outliers.

# #### replace ',' with '.' and cast to float

# In[452]:


df["S_VALUE_CONTRACT_EUR"] = df["S_VALUE_CONTRACT_EUR"].str.replace(",", ".").astype(float)
df["B_VALUE_CONTRACT_EUR"] = df["B_VALUE_CONTRACT_EUR"].str.replace(",", ".").astype(float)


# ### drop outliers in Value of Contract

# #### Function to transform positive and negative values into logs for ease of plotting

# In[ ]:


def log_transform(x):
    if x > 0:
        return np.log(x)
    if x == 0:
        return 0
    if x < 0:
        return -(np.log(abs(x)))
    else:
        return np.nan


# In[ ]:


df["S_LOG_VALUE_CONTRACT"] = df["S_VALUE_CONTRACT_EUR"].apply(log_transform)


# In[ ]:


df["B_LOG_VALUE_CONTRACT"] = df["B_VALUE_CONTRACT_EUR"].apply(log_transform)


# In[ ]:


outlier_threshold = np.log(1000000000)


# check how many values are affected

# In[473]:


val_contr_outlier = (abs(df["S_LOG_VALUE_CONTRACT"]) > outlier_threshold) | (abs(df["B_LOG_VALUE_CONTRACT"]) > outlier_threshold)


# In[475]:


df = df.loc[[not i for i in val_contr_outlier], :]
df.reset_index(drop = True, inplace = True)


# ### investigate intragroup trades

# index of intragroup trades

# In[437]:


ig_idx = np.where(df['Common_INTRAGROUP_TRADE'] == 'Y')[0]


# fraction of all

# In[438]:


sum(df['Common_INTRAGROUP_TRADE'] == 'Y')/len(df)


# In[186]:


ig_df = df.loc[ig_idx,:].copy()
ig_df.reset_index(inplace = True, drop = True)


# ## Generate unique pairs of CPs

# construct a set of distinct pairs, that is distinct combination of cps irrespective of order of buyer, seller.

# In[479]:


cp_pairs = list(zip(df["S_ID"], df["B_ID"]))


# In[481]:


set_pairs = set(cp_pairs)


# In[483]:


distinct_pairs = Counter(frozenset(pair) for pair in cp_pairs)


# extract the tuples of distinct pairs

# In[486]:


unique_pairs = []
for key, val in distinct_pairs.most_common():
    unique_pairs.append(tuple(key))


# ### Netting for directed exposure

# Procedure to generate the directed aggregate exposures for each unique pair of CPs, that is sum all the values on the respective contract side OR the gross exposure for the undirected exposures

# set false for undirected gross exposure

# In[ ]:


directed = True


# In[489]:


start = time.time()
pair_exposures = []
for s_b_pair in unique_pairs:
    # Gather all trade where the parties are involved together
    trades_idx = np.where((df['S_ID'] == s_b_pair[0]) & (df["B_ID"] == s_b_pair[1]) | ((df['B_ID'] == s_b_pair[0]) & (df["S_ID"] ==s_b_pair[1])))[0]
    trades_df = df.loc[trades_idx,:]
    # get the asset classes where the parties have transactions open, as not all trade in all classes                      
    asset_classes = set(trades_df['Common_CLASS_ASSET'])
    for asset in asset_classes:
            # Subset by asset classes
            by_asset = trades_df.loc[trades_df['Common_CLASS_ASSET'] == asset,]
            if not by_asset.empty:
                if directed:
                # build the exposure as the sum over all transaction values of the corresponding side in each trade
                    s_exposure = sum(by_asset.loc[by_asset['S_ID'] == s_b_pair[0], 'S_VALUE_CONTRACT_EUR']) + sum(by_asset.loc[by_asset['B_ID'] == s_b_pair[0], 'B_VALUE_CONTRACT_EUR'])
                    b_exposure = sum(by_asset.loc[by_asset['S_ID'] == s_b_pair[1], 'S_VALUE_CONTRACT_EUR']) + sum(by_asset.loc[by_asset['B_ID'] == s_b_pair[1], 'B_VALUE_CONTRACT_EUR'])
                    pair_exposures.append((s_b_pair[0],s_b_pair[1],s_exposure, asset))
                    pair_exposures.append((s_b_pair[1],s_b_pair[0],b_exposure, asset))
                else:
                    s_exposure = sum(abs(by_asset.loc[by_asset['S_ID'] == s_b_pair[0], 'S_VALUE_CONTRACT_EUR'])) + sum(abs(by_asset.loc[by_asset['B_ID'] == s_b_pair[0], 'B_VALUE_CONTRACT_EUR']))
                    b_exposure = sum(abs(by_asset.loc[by_asset['S_ID'] == s_b_pair[1], 'S_VALUE_CONTRACT_EUR'])) + sum(abs(by_asset.loc[by_asset['B_ID'] == s_b_pair[1], 'B_VALUE_CONTRACT_EUR']))
                    pair_exposures.append((s_b_pair[0],s_b_pair[1],s_exposure + b_exposure, asset))
                    
            else:
                  continue
end = time.time()
end - start                              


# Generate DataFrame from tuples of exposure

# In[328]:


exp_df = pd.DataFrame(pair_exposures, columns = ["CP1", "CP2", "exp", "class"])

