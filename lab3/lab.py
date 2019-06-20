
# In[]
import os
import pandas as pd
import numpy as np
import mmh3
import random
import matplotlib.pyplot as plt

# Util functions
###############################
def loadData(path):
    col_names = ['Date_1', 'Date_2', 'Durat', 'Prot', 'SrcIPAddr:Port', 'direction', 'DstAddr:Port', 'Flags', 'Tos', 'Packets', 'Bytes',
       'Flows', 'Label', 'Labels']
    df = pd.read_csv(path, delimiter='\s+', skiprows=1, names=col_names)
    df['Date'] = df['Date_1'] + ' ' + df['Date_2']
    pd.to_datetime(df['Date'], format=r'%Y-%m-%d %H:%M:%S.%f')
    df.set_index('Date', inplace=True)
    df.drop(['Date_1', 'Date_2', 'direction'], axis = 1, inplace=True)
    
    return df
    

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
DATAFILE1 = r'lab3/data/CTU-13-Scen-3.csv'

df_scen3 = loadData(DATAFILE1)
df_scen3.head()


#%%
#############################
# Resevoir sampling
#############################
sample = reservoirSample(df_scen3, 1000)
# display(sample.head())

#%%
# sample['DstAddr:Port.1'].value_counts()
sample['SrcIPAddr:Port'].value_counts()


#%%
##############################
# Sketching Task
##############################
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
###############################
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
DATAFILE_Scen10 = r'lab3/data/CTU-13-Scen-3.csv'
INFECTED_HOST = ''
df_scen10 = loadData(DATAFILE_Scen10)
df_scen10.head()

#%%
df_scen10[(df_scen10['Label'] != 'Background') & (df_scen10['Label'] != 'LEGITIMATE')].head()

#%%
###############################
# Botnet profiling
###############################

#%%
###############################
# Flow classification
###############################

#%%
###############################
# Bonus: Adversarial examples
###############################
#%%
