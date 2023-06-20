#!/usr/bin/env python
# coding: utf-8

# In[2]:


#Supervised learning algorithms are trained using labeled examples
#such as an input where the desired output is known.
#The network receives a set of inputs along with the corresponding correct outputs and 
#the algorithm learns by comparing outputs to find errors.
#It then modifies the model accordingly.
#Supervised learning is commonly used in applications where historical data predicts likely 
#future events.


# In[3]:


#The data is often split into 3 sets
#1)Training Data- Used to train model parameters
#2)Validation  Data- used to determine what model hyperparameters to adjust
#Test data- used to get final performance metric.


# In[5]:


#Model Evaluation-Classification Metrics


# In[6]:


#The key classification metrics we should understand are:
#1)Accuracy
#2)Recall
#3)Precision
#4)F1-Score


# In[7]:


#Accuracy in classification problems is the number of correct predictions
# made by the model divided by total number of predictions.
#Accuracy is not good with unbalanced classes.


# In[8]:


#Recall is the ability of a model to find ll the relevant 
#cases within a dataset.
#The precise definition of recall is the number of true positives 
#divided by the number of true positives plus the number of false negatives


# In[9]:


#Precision is the ability of a clad=ssification model to identify only the
#relevant data points.
#Precision is defned as the number of true positives divided 
#by the number of true positives plus the number of false positives. 


# In[10]:


#While recall expresses the ability to find all relevant instances in a dataset
#precision expresses the proportion of the data points our model says
#was relevant actuallly were relevant.


# In[11]:


#F1-score - In cases where we want to find an optimal blend of precision
#and recall we can combine the two metrics using F!-Score.  
#F=2*precision*recall/(precision+recall)
#The reason we use harmonic mean instead of a simple average because 
#it punishes extreme values.
#A classifier with a precision of 1.0 and a recall of 0.0
#has a simple average of 0.5 but an f1 score of 0.


# In[12]:


#Evaluating Performance-Regression


# In[13]:


#Regression is a task when a model attempts to predict continuous values.


# In[14]:


#Common evaluation metrics for regression:
#1)Mean absolute error
#2)Mean sqaured error
#3)Root mean square error


# In[ ]:




