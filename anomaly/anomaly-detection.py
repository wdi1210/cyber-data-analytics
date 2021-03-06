


# In[]:
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import math


DATAPATH = 'anomaly/data'

dateparse = lambda x: pd.datetime.strptime(x, '%d/%m/%y %H')

data3 = pd.read_csv(os.path.join(DATAPATH, 'BATADAL_dataset03.csv'), index_col=0, 
parse_dates=[0], date_parser=dateparse)
data4 = pd.read_csv(os.path.join(DATAPATH, 'BATADAL_dataset04.csv'), index_col=0, parse_dates=[0], date_parser=dateparse)
data_test = pd.read_csv(os.path.join(DATAPATH, 'BATADAL_test_dataset.csv'), index_col=0, parse_dates=[0], date_parser=dateparse)

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

def timerseries_test_train_split(data, labels, split):
    split_Nr = int(len(data) * split)
    X_train = data.iloc[:-split_Nr]
    X_test = data.iloc[-split_Nr:]
    y_train = labels.iloc[:-split_Nr]
    y_test = labels.iloc[-split_Nr:]

    return [X_train, X_test, y_train, y_test]


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

model = AR(getTankLevel(data, 1)).fit()
yhat = model.predict(len(data)-30, len(data))
actual = getTankLevel(data.tail(31), 1)

plt.plot(actual.to_numpy(), label='Actual')
plt.plot(yhat.to_numpy(), label='Predicted')
plt.legend()
plt.show()

#print(yhat.to_numpy())
#print(actual.to_numpy())


# Select Data and evaluate parameters 
# In[]:
from statsmodels.tsa.arima_model import ARMA
from pandas.plotting import autocorrelation_plot

AR_data = data4['L_T1'] # Update accordingly between L_T1 - L_T5

X_train, X_test, y_train, y_test = timerseries_test_train_split(AR_data, AR_data, 0.2)

ax = autocorrelation_plot(X_train)
ax.set_xlim([0, 35])

##################
# Train ARMA model and predict
#################
# In[]:
# we update order=(p,q) accordingly. q=1 for all.
# For L_T1: p=10, L_T2: p=6, L_T3 and LT_4: p=4, L_T5: p=3.
model = ARMA(X_train.values, order=(10, 1)).fit()

#print(model.summary())
# plot residual errors
residuals = pd.DataFrame(model.resid)
residuals.plot(title="Residuals")
plt.show()
residuals.plot(kind='kde', title='Residual Density')
plt.show()
#print(residuals.describe()) 

predictions = model.forecast(steps=len(y_test))[0]

# Compute the performance of the prediction
# In[]:
x = pd.DataFrame(data = predictions, index = y_test.index.values)
x.index.name = y_test.index.name
result = pd.concat([y_test, x], axis=1)
result.columns = ['Expected', 'Predictions']

# result.plot()

diff = result['Expected'] - result['Predictions']
# diff.plot(kind='kde')
# plt.plot(diff)

sns.distplot(diff.values, hist=False, kde=True, 
             bins=int(180/5), color = 'lightblue',
             hist_kws={'edgecolor':'black'}, label='Residual error for L_T1')

# for i in range(1, len(test)):
#     print('got {} - expected {}'.format(predictions[i], test.values[i]))


# Plotting expected and predicted values from ARMA
# In[]:
print(result)
plt.plot(result['Expected'], label='Expected')
plt.plot(result['Predictions'], label='Predicted')
plt.legend()
plt.show()

##################
# Discretize data
#################
# In[]:
dataDisc = data4[['L_T1']]

# data.plot()

# print(np.array(data).ravel())

dataDisc['group'] = pd.cut(np.array(dataDisc).ravel(), 5, labels=['very low', 'low','medium', 'high', 'very high'])

dataDisc['label'] = data4_labels

sns_plot = sns.lmplot(x='DATETIME', y='L_T1', hue='group', data=dataDisc.reset_index(), fit_reg=False)
plt.savefig("discretization-group.png")

sns_plot = sns.lmplot(x='DATETIME', y='L_T1', hue='label', data=dataDisc.reset_index(), fit_reg=False)
plt.savefig("discretization-label.png")

# print(''.join(dataDisc['group']))

# from prefixspan import PrefixSpan
# ps = PrefixSpan(''.join(dataDisc['group']))

# print(ps.frequent(7))

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

# Anomaly score by distance squared
def anomalyScores(originalDF, reducedDF):
    loss = np.sum((np.array(originalDF)-np.array(reducedDF))**2, axis=1)
    loss = pd.Series(data=loss,index=originalDF.index)
    loss = (loss-np.min(loss))/(np.max(loss)-np.min(loss))
    return loss


X_train, X_test, y_train, y_test = train_test_split(data4, data4_labels, test_size=0.33, random_state=2019)


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


# Discretized

print(dataDisc.head())

TP = dataDisc.loc[(dataDisc['label'] == 1) & (dataDisc['group'] == 'very high')]
FP = dataDisc.loc[(dataDisc['label'] != 1) & (dataDisc['group'] == 'very high')]
TN = dataDisc.loc[(dataDisc['label'] != 1) & (dataDisc['group'] != 'very high')]
FN = dataDisc.loc[(dataDisc['label'] == 1) & (dataDisc['group'] != 'very high')]

DiscPrecision = len(TP.index) / ( len(TP.index) + len(FP.index))
DiscRecall = len(TP.index) / ( len(TP.index) + len(FN.index))

print(DiscPrecision, DiscRecall)


# PCA
TP = anomalyScoresPCA.loc[(anomalyScoresPCA['ATT_FLAG'] == 1) & (anomalyScoresPCA['anomalyScore'] > 0.2)]
FP = anomalyScoresPCA.loc[(anomalyScoresPCA['ATT_FLAG']!= 1) & (anomalyScoresPCA['anomalyScore'] > 0.2)]
TN = anomalyScoresPCA.loc[(anomalyScoresPCA['ATT_FLAG']!= 1) & (anomalyScoresPCA['anomalyScore'] <= 0.2)]
FN = anomalyScoresPCA.loc[(anomalyScoresPCA['ATT_FLAG']!= 1) & (anomalyScoresPCA['anomalyScore'] <= 0.2)]

PCAprecision = len(TP.index) / ( len(TP.index) + len(FP.index))
PCArecall = len(TP.index) / ( len(TP.index) + len(FN.index))

print(PCAprecision, PCArecall)



######################
##### Bonus: PyTorch
######################

# Define Pytorch long short term memory network
# In[]:
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class LSTM(nn.Module):

    def __init__(self, input_dim, hidden_dim, batch_size, output_dim=1,
                    num_layers=2):
        super(LSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.num_layers = num_layers

        # LSTM layer
        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers)
        # Output layer
        self.linear = nn.Linear(self.hidden_dim, output_dim)
        # Loss function
        self.loss_fn = nn.MSELoss()

    # Initial hidden states
    def init_hidden(self):
        return (torch.zeros(self.num_layers, self.batch_size, self.hidden_dim),
                torch.zeros(self.num_layers, self.batch_size, self.hidden_dim))

    # Forward pass and return the prediction (single step only)
    def forward(self, input):
        lstm_out, self.hidden = self.lstm(input.view(len(input), self.batch_size, -1))
        # prediction
        y_pred = self.linear(lstm_out[-1].view(self.batch_size, -1))

        return y_pred.view(-1)

    # Compute loss
    def loss(self, pred, truth):
        return self.loss_fn(pred, truth)



# Define and train model
# In[]:

test_size = 0.2
torch_data = getTankLevel(data4, 1)
X_train, X_test, y_train, y_test = timerseries_test_train_split(torch_data, torch_data, test_size)

# Data split
X_train = torch.from_numpy(X_train).type(torch.Tensor).view([1, -1, 1])
# X_test = torch.from_numpy(stable_ar.X_test).type(torch.Tensor).view([input_size, -1, 1])
y_train = torch.from_numpy(y_train).type(torch.Tensor).view(-1)
# y_test = torch.from_numpy(stable_ar.y_test).type(torch.Tensor)


# Define model and optimiser
model = LSTM(input_dim=1, hidden_dim=64, batch_size=math.ceil((1-test_size)*len(torch_data)), output_dim=1, num_layers=2)
training_epochs = 500
optimiser = torch.optim.Adam(model.parameters())

# set training mode
model.train()
# run epochs
hist = np.zeros(training_epochs)
for t in range(training_epochs):
    # Clear stored gradient
    # model.zero_grad()
    
    # Initialise hidden state
    # Don't do this if you want your LSTM to be stateful
    model.hidden = model.init_hidden()
    
    # Forward pass
    y_pred = model.forward(X_train)

    # Compute loss
    loss = model.loss(y_pred, y_train)
    hist[t] = loss.item()

    # Zero out gradient between steps
    optimiser.zero_grad()
    # Backward pass
    loss.backward()
    # Update parameters
    optimiser.step()

# Plots and performance
# plt.plot(y_pred.detach().numpy(), label="Predictions")
# plt.plot(y_train.detach().numpy(), label="Data")
# plt.legend()
# plt.show()

plt.plot(y_pred.detach().numpy()[1600:1800], label="Predictions")
plt.plot(y_train.detach().numpy()[1600:1800], 'r--', label="Data")
plt.legend()
plt.show()

plt.plot(hist, label="Training loss")
plt.legend()
plt.show()

diff = np.subtract(y_pred.detach().numpy(), y_train.detach().numpy())[1600:1800]
plt.plot(diff, label="Difference")
plt.legend()
plt.show()



#%%

torch_df = pd.DataFrame({"predictions":y_pred.detach().numpy(), "actual" : y_train.detach().numpy()})
torch_df.head()

torch_df = pd.concat([torch_df, data4_labels], axis = 1)
torch_df.set_index("DATETIME", inplace=True)
torch_df.head()

torch_df['delta'] = (torch_df['predictions'] - torch_df['actual'])**2

torch_df.head()

sns_plot = sns.lmplot(x='DATETIME', y='delta', hue='ATT_FLAG', data=torch_df.reset_index(), fit_reg=False)

# data4_labels.reset_index().index[data4_labels['ATT_FLAG'] == 1].tolist()

#%%
