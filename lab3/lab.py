# In[]
# Util functions
###############################
import pandas as pd
import random

def modifyData(df):
    df['Date'] = df['Date_1'] + ' ' + df['Date_2']
    df.set_index('Date', inplace=True)
    df.drop(['Date_1', 'Date_2', 'direction'], axis = 1, inplace=True)
    

def reservoirSample(input_df, result_df, rezSize):
    if len(result_df) < rezSize:
        diff = rezSize - len(result_df)
        diff = min(diff, len(input_df))
        result_df = pd.concat([input_df.head(diff), result_df])
    else:
        for i in range(rezSize, len(input_df)):
            j = random.randint(1, rezSize)
            if j < rezSize:
                result_df.iloc[j] = input_df.iloc[i]

def concat2(a, b):
    return pd.concat([a,b])




# In[]
###  Resevoir samling
################################
import os
import pandas as pd

DATAFILE = r'lab3/data/capture20110812.pcap.netflow.labeled.csv'
CHUNK_SIZE = 100


col_names = ['Date_1', 'Date_2', 'Durat', 'Prot', 'SrcIPAddr:Port', 'direction', 'DstAddr:Port.1', 'Flags', 'Tos', 'Packets', 'Bytes',
       'Flows', 'Label', 'Labels']
# chunks = pd.read_csv(DATAFILE, delimiter='\s+', skiprows=1, names=col_names, chunksize=CHUNK_SIZE)
csv = pd.read_csv(DATAFILE, delimiter='\s+', skiprows=1, names=col_names)
#%%
num_chunks = int(sum(1 for row in open(DATAFILE, 'r')) / CHUNK_SIZE) + 1
chunk_id = iter(range(1, num_chunks+1))


print(num_chunks)

resevoir = pd.DataFrame()
for df in chunks:
    currentid = next(chunk_id)
    print("Processing chunk {} of {}".format(currentid, num_chunks), end="\r")

    if currentid == 2:
        break

    modifyData(df)

    display(df.head())

    # resevoir = pd.concat([df, resevoir])
    resevoir = concat2(df, resevoir)

    display(resevoir.head())
    # reservoirSample(df, resevoir, 200)
    
    
#%%
resevoir.head()

#%%
