#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import matplotlib.pyplot as plt
import pandas as pd

IN_FILE = '/home/wouter/Downloads/data_for_student_case.csv(1)/data_for_student_case.csv'


# In[ ]:


data = pd.read_csv(IN_FILE, nrows=10000)


print(data.head())
# print(len(data['card_id'].unique()))

    # data.sort_values('card_id', inplace=True)

    # print(data['card_id'].head())

    data['mail_count'] = data.groupby(['card_id'])['mail_id'].transform('count')

    # print(data.groupby(['card_id', 'mail_id']).describe())

    # d = data[['card_id', 'mail_id']].copy()

    # print(data.head())
    # d.plot()
    plt.scatter(data['card_id'], data['mail_count'])
    plt.show()

