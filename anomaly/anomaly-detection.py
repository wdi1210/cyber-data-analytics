


# In[]:
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

DATAPATH = '~/Documents/Github/cyber-data-analytics/anomaly/data/'

dateparse = lambda x: pd.datetime.strptime(x, '%d/%m/%y %H')

data3 = pd.read_csv(DATAPATH +'BATADAL_dataset03.csv', index_col=0, parse_dates=[0], date_parser=dateparse)
data4 = pd.read_csv(DATAPATH +'BATADAL_dataset04.csv', index_col=0, parse_dates=[0], date_parser=dateparse)
data_test = pd.read_csv(DATAPATH +'BATADAL_test_dataset.csv', index_col=0, parse_dates=[0], date_parser=dateparse)

# data4 and data_test seem to have spaces before the labels....
data4.columns = data3.columns
data_test.columns = data3.columns[:len(data3.columns)-1]


# for k, v in enumerate(data4.columns):
#     print("{} - {}".format(k, v))



# Helper functions

def getDate(df):
    return df['DATETIME']

# 1 to 7
def getTankLevel(df, tank_id):
    assert tank_id >= 1 and tank_id <= 7, 'Invalid tank id, must be between 1 and 7'
    return df[df.columns[tank_id+1]]

# 1 to 11
def getPumpInfo(df, pump_id):
    assert pump_id >= 1 and pump_id <= 11, 'Invalid pump id, must be between 1 and 11'
    flow = 8+pump_id
    state = 8+pump_id+2
    return df[df.columns[flow:state]]

# Only valve 2
def getValve(df):
    return df[["F_V2", "S_V2"]]

# Pressure levels, 1 to 12
def getPressure(df, junction_id):
    assert junction_id >= 1 and junction_id <= 12, 'Invalid junction id, must be between 1 and 12'
    return df[df.columns[32+junction_id:32+junction_id+1]]

def getAttackFlag(df):
    return df[['ATT_FLAG']]


data3_labels = getAttackFlag(data3)
data4_labels =  getAttackFlag(data4)

data3.drop(columns=['ATT_FLAG'], inplace=True)
data4.drop(columns=['ATT_FLAG'], inplace=True)


# In[]
# data = data3.iloc[0:500]

# tank1 = getTankLevel(data, 1)

# ma = tank1.rolling(20).sum()

# info = pd.concat([getDate(data), tank1, ma], axis = 1)
# info.columns = ['Datetime', 'L_T2', 'MA_L_T2']
# info.plot(x='Datetime')
# info.head()


##################
# Basic prediction
#################
# In[]:
from statsmodels.tsa.ar_model import AR

data = data3

model = AR(getTankLevel(data3, 1)).fit()

yhat = model.predict(len(data)-10, len(data))
print(yhat)
print(getTankLevel(data.tail(10), 1))


##################
# Select Data and evaluate parameters 
#################
# In[]:
from statsmodels.tsa.arima_model import ARMA
from pandas.plotting import autocorrelation_plot

data = data3

test_train_split = int(len(data) * 0.7)
train, test = data.iloc[0:test_train_split], data.iloc[test_train_split:len(data)]


ax = autocorrelation_plot(train)
ax.set_xlim([0, 100])


##################
# Train ARMA model and predict
#################
# In[]:
model = ARMA(train.values, order=(10, 2)).fit()

print(model.summary())
# plot residual errors
residuals = pd.DataFrame(model.resid)
residuals.plot(title="Residuals")
plt.show()
residuals.plot(kind='kde', title='Residual Density')
plt.show()
print(residuals.describe()) 

predictions = model.forecast(steps=len(test))[0]

##################
# Compute the performance of the prediction
##################
# In[]:
x = pd.DataFrame(data = predictions, index = test.index.values)
x.index.name = test.index.name
result = pd.concat([test, x], axis=1)
result.columns = ['Expected', 'Predictions']

# result.plot()

diff = result['Expected'] - result['Predictions']
# diff.plot(kind='kde')
# plt.plot(diff)

sns.distplot(diff.values, hist=False, kde=True, 
             bins=int(180/5), color = 'blue',
             hist_kws={'edgecolor':'black'})

# for i in range(1, len(test)):
#     print('got {} - expected {}'.format(predictions[i], test.values[i]))



##################
# Discretize data
#################
# In[]:
data = data4[['L_T1']]

# data.plot()

# print(np.array(data).ravel())

data['group'] = pd.cut(np.array(data).ravel(), 5, labels=['L', 'l','m', 'h', 'H'])

data['label'] = data4_labels

sns_plot = sns.lmplot(x='DATETIME', y='L_T1', hue='group', data=data.reset_index(), fit_reg=False)
plt.savefig("discretization-group.png")

sns_plot = sns.lmplot(x='DATETIME', y='L_T1', hue='label', data=data.reset_index(), fit_reg=False)
plt.savefig("discretization-label.png")



# cut_data.head()
# time = getDate(data4)
# plt.bar(data4), data)
# plt.xticks(rotation=90)


###############
# PCA anomaly detection
###############
#%%
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve, average_precision_score, roc_curve, auc
import numpy as np

def anomalyScores(originalDF, reducedDF):
    loss = np.sum((np.array(originalDF)-np.array(reducedDF))**2, axis=1)
    loss = pd.Series(data=loss,index=originalDF.index)
    loss = (loss-np.min(loss))/(np.max(loss)-np.min(loss))
    return loss


X_train, X_test, y_train, y_test = train_test_split(data4, data4_labels, test_size=0.33, random_state=2019, stratify=data4_labels)


## Evaluate the amount of PCA features to use
pca = PCA()
pca.fit(X_train)
#Plotting the Cumulative Summation of the Explained Variance
plt.figure()
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of Components')
plt.ylabel('Variance (%)') #for each component
plt.title('PCA Explained Variance')
plt.savefig('nr-of-pca.png', dpi=100)

plt.show()

n_components = 6
whiten = False
random_state = 2019

pca = PCA(n_components=n_components, whiten=whiten, 
          random_state=random_state)
X_train_PCA = pca.fit_transform(X_train)
X_train_PCA = pd.DataFrame(data=X_train_PCA, index=X_train.index)

X_train_PCA_inverse = pca.inverse_transform(X_train_PCA)
X_train_PCA_inverse = pd.DataFrame(data=X_train_PCA_inverse, 
                                   index=X_train.index)

anomalyScoresPCA = anomalyScores(X_train, X_train_PCA_inverse)
# anomalyScoresPCA.columns
anomalyScoresPCA = pd.concat([anomalyScoresPCA, data4_labels], axis=1, join='inner')



# print(anomalyScoresPCA.head())
# print('columns', anomalyScoresPCA.columns)
anomalyScoresPCA.rename(columns = {0:'anomalyScore'}, inplace=True)
# print('columnsnew', anomalyScoresPCA.columns)

# print('index', anomalyScoresPCA.index.name)

anomalyScoresPCA.reset_index(inplace=True)
print(anomalyScoresPCA.head())
sns_plot = sns.lmplot(x='DATETIME', y='anomalyScore', hue='ATT_FLAG', data=anomalyScoresPCA, fit_reg=False)
plt.savefig("pca_distance.png")

# preds = plotResults(y_train, anomalyScoresPCA, True)


##############
# Comparison
##############
# In[]:




#%%