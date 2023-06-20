#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


df=pd.read_csv('USA_Housing.csv')


# In[4]:


df.head(5)


# In[5]:


df.info()


# In[6]:


df.describe()


# In[7]:


df.columns


# In[8]:


sns.pairplot(df)


# In[9]:


sns.distplot(df['Price'])


# In[13]:


sns.heatmap(df.corr(),annot=True)


# In[11]:


df.corr()


# In[15]:


df.columns


# In[24]:


X=df[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
       'Avg. Area Number of Bedrooms', 'Area Population', 'Price']]


# In[25]:


Y=df['Price']


# In[26]:


from sklearn.model_selection import train_test_split


# In[27]:


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.4, random_state=101)


# In[28]:


from sklearn.linear_model import LinearRegression


# In[29]:


lm=LinearRegression()


# In[30]:


lm.fit(X_train,y_train)


# In[31]:


print(lm.intercept_)


# In[32]:


lm.coef_


# In[33]:


X_train.columns


# In[34]:


cdf=pd.DataFrame(lm.coef_,X.columns,columns=['Coeff'])


# In[35]:


cdf


#  Predictions
#  

# In[36]:


predictions=lm.predict(X_test)


# In[37]:


predictions


# In[38]:


y_test


# In[39]:


plt.scatter(y_test,predictions)


# In[40]:


sns.distplot((y_test-predictions))


# In[41]:


from sklearn import metrics


# In[42]:


metrics.mean_absolute_error(y_test,predictions)


# In[43]:


metrics.mean_squared_error(y_test,predictions)


# In[44]:


np.sqrt(metrics.mean_squared_error(y_test,predictions))


# # Bias Variance Trade Off

# Bias variance trade off is a fundamental topic of understanding your model's 
# performance.
# 1)The bias variance trade off is the point where we are adding just noise by adding model complexity(flexibilty).
# 2)The training error goes down as it has to,but the test error is starting to go up.
# 3)The model after the bias trade-off begins to overfit.

# In[ ]:




