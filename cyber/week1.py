#!/usr/bin/env python
# coding: utf-8

# In[3]:


import matplotlib.pyplot as plt
import pandas as pd

IN_FILE = '/home/wouter/Downloads/data_for_student_case.csv(1)/data_for_student_case.csv'


# In[5]:
data = pd.read_csv(IN_FILE)


# In[]:

# print(data.loc[data['card_id'] == 'card114487'])

count = data.groupby(['card_id'])['ip_id'].nunique()

count.plot()

# data['count'] = 

# print(data.head())



# print(data[['card_id', 'simple_journal', 'ip_count']])

# plt.figure(2)
# plt.scatter(count)
# plt.show()



# In[]:

# print(data.loc[data['card_id'] == 'card114487'])

count = data.groupby(['card_id'])['mail_id'].nunique()

count.plot()



#%%
