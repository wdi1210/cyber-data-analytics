#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.svm import SVC



IN_FILE = '/home/wouter/Downloads/data_for_student_case.csv(1)/data_for_student_case.csv'

def plot_roc(fpr, tpr):
    roc_auc = auc(fpr, tpr)
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.title('ROC Curve of kNN')
    plt.show()


# In[ ]:
data = pd.read_csv(IN_FILE)

features = pd.DataFrame()
# features['id'] = data['txid']

# Truth label
labels = data['simple_journal'].apply(lambda x : int(x == 'Chargeback'))

features['amount'] = data['amount']
features['shoppercountrycode'] = pd.factorize(data['shoppercountrycode'])[0]
features['shopperinteraction'] = pd.factorize(data['shopperinteraction'])[0]

# In[ ]:
# Split in test/train sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size = 0.2)
print("Legit/Fraud", Counter(labels))

# In[]
#SMOTE to unbalance data
sm = SMOTE(random_state=42)

X_train, y_train = sm.fit_resample(X_train, y_train)

print("Legit/Fraud", Counter(y_train))


# In[]:
#KNN Classifier
clf = KNeighborsClassifier()
clf.fit(X_train, y_train)

y_scores = clf.predict_proba(X_test)
fpr, tpr, threshold = roc_curve(y_test, y_scores[:, 1])

plot_roc(fpr, tpr)

# In[]:
#Random Forrest Classifier
clf = RandomForestClassifier()
clf.fit(X_train, y_train)

y_scores = clf.predict_proba(X_test)
fpr, tpr, threshold = roc_curve(y_test, y_scores[:, 1])

plot_roc(fpr, tpr)

# In[]:
#SVM
clf = SVC()
clf.fit(X_train, y_train)

y_scores = clf.predict_proba(X_test)
fpr, tpr, threshold = roc_curve(y_test, y_scores[:, 1])

plot_roc(fpr, tpr)


