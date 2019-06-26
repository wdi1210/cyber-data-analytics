
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
INFECTED_HOSTS = ['147.32.84.165', '147.32.84.191', '147.32.84.192', '147.32.84.193', '147.32.84.204', '147.32.84.205', '147.32.84.206', '147.32.84.207', '147.32.84.208', '147.32.84.209']

NORMAL_HOSTS = ['147.32.84.170', '147.32.84.134', '147.32.84.164', '147.32.87.36', '147.32.80.9', '147.32.87.11']
df_scen10_filtered = df_scen10[df_scen10['SrcIPAddr'] == INFECTED_HOSTS[0]]
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
toFactorize = ['Protocol', 'Flags', 'SrcIPAddr', 'SrcPort', 'DstIPAddr', 'DstPort', 'Label']
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
toFactorize = ['Protocol', 'Flags']
for tf in toFactorize:
    df_scen10_nobg[tf] = pd.factorize(df_scen10_nobg[tf])[0]

df_scen10_nobg_inf = df_scen10_nobg[df_scen10_nobg['SrcIPAddr'] == INFECTED_HOSTS[0]]
df_scen10_nobg_norm = df_scen10_nobg[df_scen10_nobg['SrcIPAddr'] == NORMAL_HOSTS[0]]

#%%
display('infected', df_scen10_nobg_inf[['Packets', 'Bytes', 'Protocol', 'Flags']].describe())
display('clean', df_scen10_nobg_norm[['Packets', 'Bytes', 'Protocol', 'Flags']].describe())
# TODO Plot these discribes 


#%%
from sklearn.cluster import KMeans

for feat in ['Bytes', 'Protocol']:
    X = df_scen10_nobg[[feat]]
    distorsions = []
    maxClusters = 10
    for k in range(2, maxClusters):
        print('computing using {} clusters'.format(k), end='\r')
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(X)
        distorsions.append(kmeans.inertia_)

    fig = plt.figure(figsize=(15, 5))
    plt.plot(range(2, maxClusters), distorsions)
    plt.xticks(range(2, maxClusters))
    plt.grid(True)
    plt.title('Elbow curve of ' + feat)
    plt.savefig('elbow_{}.png'.format(feat))


#%%

def attributeEncoding(values, clusters):
    percentiles = np.arange(0, 100, 100/clusters)
    
    percentile_border = []
    for p in percentiles:
        percentile_border.append(np.percentile(values, p))
    encoded = np.zeros(len(values))
    for i, v in enumerate(values):
        for j, p in enumerate(percentile_border):
            if v > p:
                encoded[i] = j
                pass;
    
    return encoded        

inf_byte_codes = attributeEncoding(df_scen10_nobg_inf['Bytes'].values, 6)
inf_prot_codes = attributeEncoding(df_scen10_nobg_inf['Protocol'].values, 3)
inf_combined = np.add(inf_byte_codes, inf_prot_codes)

norm_byte_codes = attributeEncoding(df_scen10_nobg_norm['Bytes'].values, 6)
norm_prot_codes = attributeEncoding(df_scen10_nobg_norm['Protocol'].values, 3)
norm_combined = np.add(norm_byte_codes, norm_prot_codes)


#%%
# Botnet profiling
###############################
from hmmlearn import hmm

infected_discretized = pd.DataFrame(data = inf_combined, index = df_scen10_nobg_inf.index)
infected_discretized.head()
#%%
# Sliding window data
# paper: BClu used 2 minutes
WINDOW_WIDTH = 10
inf_sliding_window = infected_discretized.rolling(window=WINDOW_WIDTH, min_periods=1).mean()
inf_sliding_window.head()

#%%
model = hmm.GaussianHMM(n_components=4, covariance_type="full", n_iter=100)
model.fit(inf_sliding_window)

model_likelyhood = model.score(inf_sliding_window)



#%%
# Evaluate
def scoreForIP(ip):
    df_scen10_nobg_ip = df_scen10_nobg[df_scen10_nobg['SrcIPAddr'] == ip]

    if len(df_scen10_nobg_ip) == 0 :
        return np.nan

    byte_codes = attributeEncoding(df_scen10_nobg_ip['Bytes'].values, 6)
    prot_codes = attributeEncoding(df_scen10_nobg_ip['Protocol'].values, 3)
    combined = np.add(byte_codes, prot_codes)
    discretized = pd.DataFrame(data = combined, index = df_scen10_nobg_ip.index)
    sliding = discretized.rolling(window=WINDOW_WIDTH, min_periods=1).mean()

    return model.score(sliding)


#%%


TP, FP, TN, FN = 0 ,0 ,0 ,0
for inf in range(1, len(INFECTED_HOSTS)):
    score = scoreForIP(INFECTED_HOSTS[inf])
    if not np.isnan(score):
        if abs(model_likelyhood-score) > model_likelyhood:
            FN += 1
        else:
            TP += 1
    else:
        print('inf nan: {}'.format(inf))

for norm in NORMAL_HOSTS:
    score = scoreForIP(norm)
    if not np.isnan(score):
        if abs(model_likelyhood-score) > model_likelyhood:
            TN += 1
        else:
            print(norm, score)
            FP += 1
    else:
        print('norm nan: {}'.format(norm))

print('TP:{}\t FP:{}\nFN:{}\tTN:{}'.format(TP, FP, FN, TN))




#%%
# Flow classification
###############################


from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from imblearn.over_sampling import SMOTE

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import cross_val_score, cross_val_predict, cross_validate
from sklearn.metrics import roc_curve, auc, precision_recall_curve, confusion_matrix, accuracy_score, recall_score, precision_score

def plot_roc(fpr, tpr):
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.grid(True)
    plt.show()

def plot_PRcurve(recall, precision):
    roc_auc = auc(recall, precision)
    plt.plot(recall, precision, 'r', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower left')
    plt.ylabel("Precision")
    plt.xlabel("Recall")
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.title("Precision-Recall curve")
    plt.grid(True)
plt.show()

def eval_classifier(clf, X_train, X_test, y_train, y_test):
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)

    # ROC
    fpr, tpr, _ = roc_curve(y_test, predictions)
    plot_roc(fpr, tpr)
    
    # PR
    precision, recall, _ = precision_recall_curve(y_test, predictions)
    plot_PRcurve(recall, precision)
    
    # Confusion Matrix
    print("TN", "FP")
    print("FN", "TP")
    confmat = confusion_matrix(y_test, predictions.astype(int))
    print(pd.DataFrame(confmat))
    measures = {'Accuracy': accuracy_score,
                'Precision': precision_score,
                'Recall': recall_score}
    for name in measures:
        print(name, measures[name](y_test, predictions.astype(int)))



#%%
# Packet level classification

# class imbalance check
df_scen10_nobg_host['Label'].value_counts().plot.bar()

#%%
labels = df_scen10_nobg['Label'].apply(lambda x : 1 if int(x == 'Botnet') else 0)
features = df_scen10_nobg[['Duration', 'Protocol', 'Flags', 'Packets', 'Bytes', 'SrcIPAddr', 'SrcPort', 'DstIPAddr', 'DstPort']]
toFactorize = ['SrcIPAddr', 'SrcPort', 'DstIPAddr', 'DstPort']
for tf in toFactorize:
    features[tf] = pd.factorize(features[tf])[0]


#%%
#  basic eval of classifier
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size = 0.4)
eval_classifier(RandomForestClassifier(n_estimators=20), X_train, X_test, y_train, y_test)


#%%
skf = StratifiedKFold(n_splits=10, shuffle=True)
split, totalTN, totalFP, totalFN, totalTP = 0,0,0,0,0
for train_index, test_index in skf.split(features, labels):
    split += 1
    #Select the data for this round
    X_train, X_test = features.iloc[train_index], features.iloc[test_index]    
    y_train, y_test = labels[train_index], labels[test_index]
    
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
# Host level classification

# To get information on a host level we aggregate by SrcIPAddr, and 
df_scen10_nobg_host = df_scen10_nobg.groupby(['SrcIPAddr']).agg({'Duration':'mean', 'Protocol':'nunique', 'Flags':'nunique', 'Packets':'mean', 'Bytes':'mean', 'SrcPort':'nunique', 'DstIPAddr':'nunique', 'DstPort':'nunique', 'Label':'unique'})
df_scen10_nobg_host['MostProtocol'] = df_scen10_nobg.groupby('SrcIPAddr')['Protocol'].apply(lambda x: x.value_counts().index[0])
# df_scen10_nobg_host['MostSrcPort'] = df_scen10_nobg.groupby('SrcIPAddr')['SrcPort'].apply(lambda x: x.value_counts().index[0])
# df_scen10_nobg_host['MostDstPort'] = df_scen10_nobg.groupby('SrcIPAddr')['DstPort'].apply(lambda x: x.value_counts().index[0])

df_scen10_nobg_host['Label_size'] = df_scen10_nobg_host['Label'].apply(lambda x : len(x))

df_scen10_nobg_host = df_scen10_nobg_host[df_scen10_nobg_host['Label_size'] == 1]
df_scen10_nobg_host['Label'] = df_scen10_nobg_host['Label'].apply(lambda x : x[0])

df_scen10_nobg_host.head()

#%%
# class imbalance check
df_scen10_nobg_host['Label'].value_counts().plot.bar()


#%%


#%%
# Bonus: Adversarial examples
###############################
