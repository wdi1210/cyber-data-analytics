#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import pandas as pd

IN_FILE = '/home/wouter/Downloads/data_for_student_case.csv(1)/data_for_student_case.csv'


# In[ ]:
data = pd.read_csv(IN_FILE, nrows=1000)

features = pd.DataFrame()
features['id'] = data['txid']

# Truth label
features['label'] = data['simple_journal'].apply(lambda x : int(x == 'Chargeback'))

features['amount'] = data['amount']
features['shoppercountrycode'] = pd.factorize(data['shoppercountrycode'])[0]
features['shopperinteraction'] = pd.factorize(data['shopperinteraction'])[0]


print(features.head())
print(features.tail())

print("Legit", len(features.loc[features['label'] == 0]))
print("Fraud", len(features.loc[features['label'] == 1]))

# In[ ]:



