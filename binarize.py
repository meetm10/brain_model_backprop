#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pickle
import cv2


# In[2]:


file = open('mnist_data.pkl','rb')
data = pickle.load(file)


# In[3]:


train_in,train_out,test_in,test_out = data['train_input'],data['train_output'],data['test_input'],data['test_output']


# In[4]:


train_list = []
for i,t in enumerate(train_in):
    _,thresh = cv2.threshold(t,127,255,cv2.THRESH_BINARY)
    train_list.append(thresh)


# In[7]:


train_in = np.asarray(train_list)


# In[11]:


test_list = []
for i,t in enumerate(test_in):
    _,thresh = cv2.threshold(t,127,255,cv2.THRESH_BINARY)
    test_list.append(thresh)


# In[15]:


test_in = np.asarray(test_list)


# In[19]:


binarize_data = {'train_input':train_in,'train_output':train_out,'test_input':test_in,'test_output':test_out}


# In[20]:


file = open('binarized_mnist_data.pkl','wb')
pickle.dump(binarize_data,file)

