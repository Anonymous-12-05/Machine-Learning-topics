#!/usr/bin/env python
# coding: utf-8

# # K Means Clustering 
It is an unsupervised learning algo that will attempt to group similar clusters together in your data.
A typical cluster problem looks like:
    1)Cluster similar documents.
    2)Cluster Customers based on Features.
    3)Market Segmentation.
    4)identify similar physical groups.  The K Means Algorithm
1)Choose number of clusters 'K'
2)Randomly assign each point to a cluster.
3)Until clusters stop changing,repeat the following:
    3.1)For each cluster,compute the cluster centroid by taking the mean vector
        of points in the cluster.
    3.2)Assign each data point to the cluster for which the centroid is the 
        closest.
    
    
    There is no easy answer for choosing a best 'K' value
One way is the elbow method.
First of all,compute the sum of sqaured error(SSE) for some values of k.

SSE is defined as the sum of the squared distance between each member of the cluster and its centroid.
# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


from sklearn.datasets import make_blobs


# In[4]:


data=make_blobs(n_samples=200,n_features=2,centers=4,cluster_std=1.8,
                random_state=101)


# In[5]:


data[0].shape


# In[7]:


data


# In[ ]:





# In[11]:


plt.scatter(data[0][:,0],data[0][:,1],c=data[1],cmap='rainbow')


# In[14]:


from sklearn.cluster import KMeans


# In[16]:


kmeans=KMeans(n_clusters=4)


# In[17]:


kmeans.fit(data[0])


# In[18]:


kmeans.cluster_centers_


# In[19]:


kmeans.labels_


# In[20]:


data[1]


# In[21]:


fig,(ax1,ax2)=plt.subplots(1,2,sharey=True,figsize=(10,6))
ax1.set_title('K Means')
ax1.scatter(data[0][:,0],data[0][:,1],c=kmeans.labels_,cmap='rainbow')
ax2.set_title('Original')
ax2.scatter(data[0][:,0],data[0][:,1],c=data[1],cmap='rainbow')


# In[ ]:




