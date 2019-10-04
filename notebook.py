#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import libraries
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

#import models
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC



# In[2]:


#load data
data = pd.read_csv('train.csv')


# In[3]:


#show top few rows
data.head(5)


# In[4]:


#visualizing data
v = data.iloc[4, 1:].values    #data.iloc[<row selection>, <column selection>]
v = v.reshape(28, 28).astype('uint8')
plt.imshow(v)


# In[5]:


#data preprocessing
df_X = data.iloc[:, 1:]
df_y = data.iloc[:, 0]
train_X, val_X, train_y, val_y = train_test_split(df_X, df_y, test_size=0.2, random_state=21)


# In[6]:


#checking features and target values
train_X.head()


# In[7]:


train_y.head()


# In[ ]:


#defining and fitting models
rf_model=RandomForestClassifier(n_estimators=100, random_state=21, verbose=0)
rf_model.fit(train_X, train_y)

#svc_model=SVC(C=10000, gamma='auto', verbose = 1)
print("sv")
svc_model=SVC()
svc_model.fit(train_X, train_y)


# In[ ]:


#accuracy of models
rf_predictions = rf_model.predict(val_X)
rf_acc = accuracy_score(val_y, rf_predictions) * 100
print('Accuracy for RandomForest: {}'.format(rf_acc))

svc_predictions = svc_model.predict(val_X)
svc_acc = accuracy_score(val_y, svc_predictions) * 100
print('Accuracy for SVC: {}'.format(svc_acc))


# In[ ]:




