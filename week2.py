#!/usr/bin/env python
# coding: utf-8

# In[1]:
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from collections import Counter

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC

from sklearn.model_selection import cross_val_score, cross_val_predict, cross_validate

from sklearn.metrics import roc_curve, auc, precision_recall_curve, confusion_matrix



IN_FILE = '/home/wouter/Downloads/data_for_student_case.csv(1)/data_for_student_case.csv'

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



def eval_classifier(clf):
    clf.fit(X_train, y_train)
    # predictions = clf.predict(X_test)
    y_scores = clf.predict_proba(X_test)
    predictions = y_scores[:,1]
    # predictions = cross_val_predict(clf, X_test, y_test, cv=10)

    # ROC
    fpr, tpr, _ = roc_curve(y_test, predictions)
    plot_roc(fpr, tpr)

    # PR
    precision, recall, _ = precision_recall_curve(y_test, predictions)
    plot_PRcurve(recall, precision)

    # Confusion Matrix

    print("TN", "FP")
    print("FN", "TP")

    print(pd.DataFrame(confusion_matrix(y_test, predictions.astype(int))))

# In[ ]:
data = pd.read_csv(IN_FILE)

features = pd.DataFrame()
# features['id'] = data['txid']

# Truth label- 1 = fraud, 0 = valid
labels = data['simple_journal'].apply(lambda x : 1 if int(x == 'Chargeback') else 0)
# labels = data['simple_journal'].apply(lambda x : 'fraud' if int(x == 'Chargeback') else 'valid')



features['amount'] = data['amount']
features['currencycode'] = pd.factorize(data['currencycode'])[0]
features['issuercountrycode'] = pd.factorize(data['issuercountrycode'])[0]
features['shoppercountrycode'] = pd.factorize(data['shoppercountrycode'])[0]
features['shopperinteraction'] = pd.factorize(data['shopperinteraction'])[0]
features['txvariantcode'] = pd.factorize(data['txvariantcode'])[0]
features['cardverificationcodesupplied'] = pd.factorize(data['cardverificationcodesupplied'])[0]
features['cvcresponsecode'] = pd.factorize(data['cvcresponsecode'])[0]
features['accountcode'] = pd.factorize(data['accountcode'])[0]
features['mail_id'] = pd.factorize(data['mail_id'])[0]
features['ip_id'] = pd.factorize(data['ip_id'])[0]
features['card_id'] = pd.factorize(data['card_id'])[0]


# In[ ]:
# Split in test/train sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size = 0.2)
print("Legit/Fraud", Counter(labels))

# In[]
#SMOTE to balance data
sm = SMOTE(random_state=42)

X_train, y_train = sm.fit_resample(X_train, y_train)

print("Legit/Fraud", Counter(y_train))

# In[]:
#KNN Classifier
print('KNN')
clf = KNeighborsClassifier()
eval_classifier(clf)

# In[]:
#DecisionTree Classifier
print('DecisionTree')
clf = DecisionTreeClassifier()
eval_classifier(clf)

# In[]:
#Random Forrest Classifier
print('RandomForest')
clf = RandomForestClassifier(n_estimators=50)
eval_classifier(clf)

# In[]:
#AdaBoost Classifier
print('AdaBoost')
clf = AdaBoostClassifier()
eval_classifier(clf)

# # In[]:
# #SVM
# clf = SVC(kernel='linear')
# eval_classifier(clf)


# # In[]:
# #OvR
# clf = OneVsRestClassifier(SVC(kernel='linear', probability=True, class_weight='balanced'), n_jobs=-1)
# eval_classifier(clf)



#%%
