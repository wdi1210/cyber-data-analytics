


# In[]:
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

DATAPATH = '~/Documents/Github/cyber-data-analytics/anomaly/data/'

data3 = pd.read_csv(DATAPATH +'BATADAL_dataset03.csv', index_col=0, parse_dates=[0])
data4 = pd.read_csv(DATAPATH +'BATADAL_dataset04.csv', parse_dates=[0], index_col=0)
data_test = pd.read_csv(DATAPATH +'BATADAL_test_dataset.csv', parse_dates=[0], index_col=0)
data4.columns = data3.columns
data_test.columns = data3.columns[:-1]

# for k, v in enumerate(data4.columns):
#     print("{} - {}".format(k, v))

def getDate(df):
    return df['DATETIME']

# 1 to 7
def getTankLevel(df, tank_id):
    return df[df.columns[tank_id+1]]

# 1 to 11
def getPumpInfo(df, pump_id):
    flow = 8+pump_id
    state = 8+pump_id+2
    return df[df.columns[flow:state]]

# Only valve 2
def getValve(df):
    return df[["F_V2", "S_V2"]]

# Pressure levels, 1 to 12
def getPressure(df, junction_id):
    return df[df.columns[32+junction_id:32+junction_id+1]]

def getAttackFlag(df):
    return df[['ATT_FLAG']]


# In[]
# data = data3.iloc[0:500]

# tank1 = getTankLevel(data, 1)

# ma = tank1.rolling(20).sum()

# info = pd.concat([getDate(data), tank1, ma], axis = 1)
# info.columns = ['Datetime', 'L_T2', 'MA_L_T2']
# info.plot(x='Datetime')
# info.head()

# In[]:

from statsmodels.tsa.ar_model import AR

data = data3

model = AR(getTankLevel(data, 1)).fit()

yhat = model.predict(len(data)-5, len(data))
print(yhat)


# In[]:
from statsmodels.tsa.arima_model import ARMA
from pandas.plotting import autocorrelation_plot

data = data3[['F_PU1']]

test_train_split = int(len(data) * 0.7)
test, train = data.iloc[0:test_train_split], data.iloc[test_train_split:len(data)]

autocorrelation_plot(train)

model = ARMA(train.values, order=(2, 5)).fit()

# print(model.summary())
# # plot residual errors
# residuals = pd.DataFrame(model.resid)
# residuals.plot(title="Residuals")
# plt.show()
# residuals.plot(kind='kde', title='Residual Density')
# plt.show()
# print(residuals.describe()) 

predictions = model.forecast(steps=len(test))[0]

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

#%%
