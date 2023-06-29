#!/usr/bin/env python
# coding: utf-8

# 
# # Support Vector Machines
SVMs are supervised learning models with associated learning algorithms that analyse data and recognise patterns,used for classification and regression analysis.
# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


from sklearn.datasets import load_breast_cancer


# In[3]:


cancer=load_breast_cancer()


# In[4]:


cancer.keys()


# In[5]:


print(cancer['DESCR'])


# In[6]:


df_feat=pd.DataFrame(cancer['data'],columns=cancer['feature_names'])


# In[7]:


df_feat.head(5)


# In[8]:


cancer['target_names']


# In[14]:


from sklearn.model_selection import train_test_split


# In[16]:


X=df_feat
y=cancer['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)


# In[17]:


from sklearn.svm import SVC


# In[18]:


model=SVC()


# In[19]:


model.fit(X_train,y_train)


# In[27]:


SVC.get_params(model)


# In[20]:


predictions=model.predict(X_test)


# In[21]:


from sklearn.metrics import classification_report,confusion_matrix


# In[22]:


print(confusion_matrix(y_test,predictions))
print('\n')
print(classification_report(y_test,predictions))


# In[24]:


from sklearn.model_selection import GridSearchCV


# In[28]:


param_grid={'C':[0.1,1,10,1000],'gamma':[1,0.1,0.001,0.0001]}


# In[29]:


grid=GridSearchCV(SVC(),param_grid,verbose=3)


# In[30]:


grid.fit(X_train,y_train)


# In[31]:


grid.best_params_


# In[32]:


grid.best_estimator_


# In[33]:


grid_predictions=grid.predict(X_test)


# In[35]:


print(confusion_matrix(y_test,grid_predictions))
print('\n')
print(classification_report(y_test,grid_predictions))


# In[ ]:




