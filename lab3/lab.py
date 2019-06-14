
# In[]
###  Resevoir samling
################################
import os
import pandas as pd

DATAFILE = r'lab3/data/capture20110812.pcap.netflow.labeled.csv'

col_names = ['Date_1', 'Date_2', 'Durat', 'Prot', 'SrcIPAddr:Port', 'direction', 'DstAddr:Port.1', 'Flags', 'Tos', 'Packets', 'Bytes',
       'Flows', 'Label', 'Labels']
csv = pd.read_csv(DATAFILE, delimiter='\s+', skiprows=1, names=col_names)


# In[]
# Util functions
###############################
import pandas as pd
import random

def modifyData(df):
    df['Date'] = df['Date_1'] + ' ' + df['Date_2']
    df.set_index('Date', inplace=True)
    df.drop(['Date_1', 'Date_2', 'direction'], axis = 1, inplace=True)
    

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
#############################
# Resevoir sampling
#############################
sample = reservoirSample(csv, 1000)
# display(sample.head())

#%%
# sample['DstAddr:Port.1'].value_counts()
sample['SrcIPAddr:Port'].value_counts()


#%%
##############################
# Sketching Task
##############################


#%%
###############################
# Flow Data Discretization
###############################

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
