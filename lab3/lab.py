
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
    df.drop(['Date_1', 'Date_2', 'direction', 'DstIPAddr:Port', 'SrcIPAddr:Port', 'Labels', 'Tos', 'Flows'], axis = 1, inplace=True)
    
    return df


#%%
# Load dataset Scen 10
#############################
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
DATAFILE1 = r'data/CTU-13-Scen-10-2.csv'

df_scen10 = loadData(DATAFILE1)
# df_scen3.head()

#%%
INFECTED_HOST = '147.32.84.165' 
df_scen10_filtered = df_scen10[df_scen10['SrcIPAddr'] == INFECTED_HOST]
display('df_scen10_filtered rows: {}'.format(len(df_scen10_filtered)))

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
sample100 = reservoirSample(df_scen10_filtered, 100)
sample1000 = reservoirSample(df_scen10_filtered, 1000)
sample5000 = reservoirSample(df_scen10_filtered, 5000)

#%%
display('original', df_scen10_filtered['DstIPAddr'].value_counts().head(10)/len(df_scen10_filtered))
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

    def get(self, key):
        values = list()
        for i in range(self.__depth):
            val = self._hash_function(key, seed=i, signed=False) % self.__width
            values.append(self._bins[val ][i])
        return min(values)


cms = CountMinSketch(1000, 10, mmh3.hash)

totalLen = len(df_scen10_filtered)
for index, row in df_scen10_filtered.iterrows():
    # if (index % 10000) == 0:
    #     print('{}/{}'.format(index, totalLen), end='\r')
    cms.add(row['DstIPAddr'])

#%%
for ip in df_scen10_filtered['DstIPAddr'].value_counts().head(10).index:
    minval = cms.get(ip)
    print('{} \t {} \t {}'.format(ip, minval, minval/totalLen))



#%%
# Flow Data Discretization
###############################
df_scen10_nobg = df_scen10[(df_scen10['Label'] != 'Background')]

#%%
import seaborn as sns

corr_mat = df_scen10_nobg.reset_index().drop(['Date'], axis=1)
toFactorize = ['Protocol', 'Flags', 'SrcIPAddr', 'SrcPort', 'DstIPAddr', 'DstPort']
for tf in toFactorize:
    corr_mat[tf+'_fact'] = pd.factorize(corr_mat[tf])[0]
corr = corr_mat.corr()

cmap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(corr, cmap=cmap,
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)
plt.savefig('flow_conf_mat.png')
plt.show()

#%%
# TODO lots showing difference between infected and normal bytes/packets and protocol/flags




#%%
from sklearn.cluster import KMeans


X = df_scen10_nobg[['Bytes', 'Packets']]
distorsions = []
for k in range(2, 20):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(X)
    distorsions.append(kmeans.inertia_)

fig = plt.figure(figsize=(15, 5))
plt.plot(range(2, 20), distorsions)
plt.xticks(range(2, 20))
plt.grid(True)
plt.title('Elbow curve of SrcPort and Packets combination')


#%%



df_scen10_nobg['relevant'] =  df_scen10_nobg[['SrcPort', 'Packets']].apply(lambda x: ':'.join(x.astype(str).values.tolist()), axis=1)
df_scen10_nobg['relevant_fact'] = pd.factorize(df_scen10_nobg['relevant'])[0] 

#%%
from saxpy.sax import sax_via_window
paa = 10
alphabet = 20
sax = sax_via_window(df_scen10_nobg['relevant_fact'].values, win_size=49, paa_size=paa, alphabet_size=alphabet, nr_strategy='none')

#%%
# Visualize discretization
discrete = []

for i in range(0,len(df_scen10_nobg),paa):
    for gram in list(sax.keys()):
        if i in sax[gram]:
            for k in range(0,paa):
                discrete.append(gram[0])
            break
   
plt.plot(np.array(discrete)[0:1000])
plt.savefig('discretization.png')
plt.show()



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
Z2 = model.predict(df_scen10)

#%%
# Flow classification
###############################
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from imblearn.over_sampling import SMOTE
from collections import Counter
from sklearn.ensemble import RandomForestClassifier


#%%
# Packet level classification

# skf = StratifiedKFold(n_splits=10, shuffle=True)
split, totalTN, totalFP, totalFN, totalTP = 0,0,0,0,0
# for train_index, test_index in skf.split(features, labels):
    # split += 1
    #Select the data for this round
    # X_train, X_test = features.loc[train_index], features.loc[test_index]    
    # y_train, y_test = labels[train_index], labels[test_index]

#Select the data for this round
X_train, X_test = features.loc[train_index], features.loc[test_index]    
y_train, y_test = labels[train_index], labels[test_index]

#Use SMOTE on training data to get balanced data
sm = SMOTE(random_state=42)
X_train, y_train = sm.fit_resample(X_train, y_train)

# Train classifier and predict
clf = RandomForestClassifier(n_estimators=20)

clf.fit(X_train, y_train)
predictions = clf.predict(X_test)

#Sum the confusion matrix values
tn, fp, fn, tp = confusion_matrix(y_test, predictions.astype(int)).ravel()
print('Split {}/10: \ntn {}\t fp {}\nfn {}\t\t tp {}'.format(split, tn, fp, fn, tp))
totalTN += tn
totalFP += fp
totalFN += fn
totalTP += tp

print("--- Total sum confmat ---")
print('tn {}\t fp {}\nfn {}\t\t tp {}'.format(totalTN, totalFP, totalFN, totalTP))

accuracy = (totalTP + totalTN) / (totalTP + totalTN + totalFN + totalFP)
precision = totalTP / (totalTP + totalFP)
recall = totalTP / (totalTP + totalFN)

print('accuracy\t{}\nprecision\t{}\nrecall\t{}'.format(accuracy, precision, recall))

#%%
# Ip level classification

#%%
# Bonus: Adversarial examples
###############################
