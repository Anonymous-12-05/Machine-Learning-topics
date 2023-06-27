#!/usr/bin/env python
# coding: utf-8

# # Trees
# 
In trees we have:
Node:
    Split for a value of a certain attribute
Edges:
    Outcome of a split to nect node
Root:
    The node that performs the first split
Leaves:
    Terminal nodes that predict the outcome
Entropy and information gain are the best mathematical methods for choosing the 
best split.
Entropy(H(S))=-Sigma p(i)(S)log2p(i)(S)
Info gain(IG(S,A))=H(S)-sigma |S(v)|*H(S(v))/SRandom Forests
To improve performance we can use many trees with a random sample of features 
chosen as the split.

.)A new random sample of features is chosen for every single tree at every single split.
.)For classification,m is typically chosen to be the square root of p.
# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


df=pd.read_csv('kyphosis.csv')


# In[3]:


df.head()


# In[4]:


df.info()


# In[5]:


sns.pairplot(df,hue='Kyphosis')


# In[6]:


from sklearn.model_selection import train_test_split


# In[7]:


X=df.drop('Kyphosis',axis=1)
y=df['Kyphosis']


# In[8]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)


# In[9]:


from sklearn.tree import DecisionTreeClassifier


# In[10]:


dtree=DecisionTreeClassifier()


# In[11]:


dtree.fit(X_train,y_train)


# In[12]:


predictions=dtree.predict(X_test)


# In[13]:


from sklearn.metrics import classification_report,confusion_matrix


# In[15]:


print(confusion_matrix(y_test,predictions))
print('\n')
print(classification_report(y_test,predictions))


# In[16]:


from sklearn.ensemble import RandomForestClassifier


# In[17]:


rfc=RandomForestClassifier(n_estimators=200)


# In[18]:


rfc.fit(X_train,y_train)


# In[19]:


rfc


# In[20]:


rfc_pred=rfc.predict(X_test)


# In[21]:


print(confusion_matrix(y_test,rfc_pred))
print('\n')
print(classification_report(y_test,rfc_pred))


# In[ ]:




