#!/usr/bin/env python
# coding: utf-8

# # Principal Component Analysis
It is an unsupervised statistical technique used to examine the interrelations among a set of variables in order to identify the underlying structure of thise variables. It is known sometimes as s general factor analysis. Where regression determines a line of best fit to a data set,factor analysis determines several orthogonal lines of best fit to the data set.

Orthogonal means "at right angles".
    Actually the lines are perpendicular to each other in n-dimensional space.
n-dimensional space is the variable sample space.
        There are as many dimensions as there are variables,so in a set with 4         variables the sample space is 4 Dimensional.Components are a linear transformation that chooses a variable system for the dataset such that the greatest variance of the data set comes to lie on the first axis.
The second greatest variance on the second axis nd so on..
This process allows us to reduce the number of variables used in an analysis.
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


# In[7]:


df=pd.DataFrame(cancer['data'],columns=cancer['feature_names'])


# In[8]:


df.head()


# In[9]:


cancer['target']


# In[10]:


from sklearn.preprocessing import StandardScaler


# In[11]:


scaler=StandardScaler()


# In[12]:


scaler.fit(df)


# In[13]:


scaled_data=scaler.transform(df)


# In[14]:


#PCA
from sklearn.decomposition import PCA


# In[15]:


pca=PCA(n_components=2)


# In[16]:


pca.fit(scaled_data)


# In[17]:


x_pca=pca.transform(scaled_data)


# In[19]:


scaled_data.shape


# In[20]:


x_pca.shape


# In[24]:


plt.figure(figsize=(8,6))
plt.scatter(x_pca[:,0],x_pca[:,1],c=cancer['target'],cmap='plasma')
plt.xlabel("First principal Component")
plt.ylabel("Second principal Component")


# In[25]:


pca.components_


# In[26]:


df_comp=pd.DataFrame(pca.components_,columns=cancer['feature_names'])


# In[27]:


sns.heatmap(df_comp,cmap="plasma")


# In[ ]:




