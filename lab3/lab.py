
# In[]
# General Imports & Util functions
###############################
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def loadData(path, nrows = None):
    col_names = ['Date_1', 'Date_2', 'Duration', 'Protocol', 'SrcIPAddr:Port', 'direction', 'DstIPAddr:Port', 'Flags', 'Tos', 'Packets', 'Bytes',
       'Flows', 'Label', 'Labels']
    df = pd.read_csv(path, delimiter='\s+', skiprows=1, names=col_names, nrows=nrows)
    df['Date'] = df['Date_1'] + ' ' + df['Date_2']
    df[['SrcIPAddr', 'SrcPort']] = df['SrcIPAddr:Port'].str.split(":", n=1, expand=True)
    df[['DstIPAddr', 'DstPort']] = df['DstIPAddr:Port'].str.split(":", n=1, expand=True)
    pd.to_datetime(df['Date'], format=r'%Y-%m-%d %H:%M:%S.%f')
    df.set_index('Date', inplace=True)
    df.drop(['Date_1', 'Date_2', 'direction', 'DstIPAddr:Port', 'SrcIPAddr:Port'], axis = 1, inplace=True)
    
    return df


#%%
# Load dataset Scen 3 (now 10....)
#############################
"""
- Infected hosts
    - 147.32.84.165:
- Normal hosts:
    - 147.32.84.170 
    - 147.32.84.134 
    - 147.32.84.164 
    - 147.32.87.36  This normal host is not so reliable since is a webserver)
    - 147.32.80.9 This normal host is not so reliable since is a dns server)
    - 147.32.87.11 This normal host is not so reliable since is a matlab server)
    """
DATAFILE1 = r'lab3/data/CTU-13-Scen-10.csv'

df_scen3 = loadData(DATAFILE1)
# df_scen3.head()

INFECTED_HOST = '147.32.84.165' 
df_scen3_filtered = df_scen3[df_scen3['SrcIPAddr'] == INFECTED_HOST]
display('df_scen3_filtered rows: {}'.format(len(df_scen3_filtered)))

#%%
# Resevoir sampling
#############################
import random

def reservoirSample(input_df, size):
    result_df = pd.DataFrame()
    for i in range(size):
        result_df = result_df.append(input_df.iloc[i])

    for i in range(size+1, len(input_df)):
        j = random.randint(0, len(input_df))
        if j < size:
            result_df.iloc[j] = input_df.iloc[i]
    return result_df
#%%
sample100 = reservoirSample(df_scen3_filtered, 100)
sample1000 = reservoirSample(df_scen3_filtered, 1000)
sample5000 = reservoirSample(df_scen3_filtered, 5000)

#%%
display('original', df_scen3_filtered['DstIPAddr'].value_counts().head(10)/len(df_scen3_filtered))
display('sample100', sample100['DstIPAddr'].value_counts().head(10)/100)
display('sample1000', sample1000['DstIPAddr'].value_counts().head(10)/1000)
display('sample5000', sample5000['DstIPAddr'].value_counts().head(10)/5000)


#%%
# Sketching Task
##############################
import mmh3

class CountMinSketch():

    def __init__(self, width, depth, hash_function):        
           self.__width = width
           self.__depth = depth
           self._bins = np.zeros((width, depth))
           self._hash_function = hash_function

    def add(self, key):
        values = list()
        for i in range(self.__depth):
            val = self._hash_function(key, seed=i, signed=False) % self.__width
            self._bins[val ][i] += 1
            values.append(val)
        return values

cms = CountMinSketch(1000, 10, mmh3.hash)

for index, row in csv.head(100000).iterrows():
    if (index % 10000) == 0:
        print('{}/{}'.format(index, len(csv), end='\r'))
    cms.add(row['SrcIPAddr:Port'])
#%%
display(cms._bins)

#%%
# Flow Data Discretization
###############################
"""
- Infected hosts
    - 147.32.84.165: 
    - 147.32.84.191: 
    - 147.32.84.192: 
    - 147.32.84.193: 
    - 147.32.84.204:
    - 147.32.84.205: 
    - 147.32.84.206:
    - 147.32.84.207: 
    - 147.32.84.208: 
    - 147.32.84.209: 
- Normal hosts:
    - 147.32.84.170 
    - 147.32.84.134
    - 147.32.84.164 
    - 147.32.87.36 This normal host is not so reliable since is a webserver)
    - 147.32.80.9 This normal host is not so reliable since is a dns server)
    - 147.32.87.11 This normal host is not so reliable since is a matlab server)
    """
DATAFILE_Scen10 = r'lab3/data/CTU-13-Scen-10.csv'
INFECTED_HOST = '147.32.84.165'
df_scen10 = loadData(DATAFILE_Scen10)
display('rows: {}'.format(len(df_scen10)))
# df_scen10.head()

#%%
from saxpy.sax import sax_via_window
df_scen10_filtered = df_scen10[(df_scen10['Label'] != 'Background') & (df_scen10['Label'] != 'LEGITIMATE') & (df_scen10['SrcIPAddr'] == INFECTED_HOST)]

sax = sax_via_window(df_scen10_filtered['Duration'].values, win_size=49, paa_size=8, alphabet_size=3, nr_strategy='none')
#%%
# Botnet profiling
###############################
from hmmlearn import hmm

# Separate in time windows
# paper: BClus used 2 minutes
WINDOW_WIDTH = '2m'
df_scen10.rolling(WINDOW_WIDTH)

# Aggregate by source IP
# paper used 1 minute
AGGREGATE_WIDTH = '1m'


# Clustering



model = hmm.GaussianHMM(n_components=3, covariance_type="full", n_iter=100)
model.fit()
Z2 = model.predict(X)

#%%
# Flow classification
###############################

#%%
# Bonus: Adversarial examples
###############################
#%%
