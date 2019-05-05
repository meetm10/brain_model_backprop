#!/usr/bin/env python
# coding: utf-8

# In[12]:


import tensorflow as tf
import numpy as np
import pickle


# In[14]:


train,test = tf.keras.datasets.mnist.load_data()


# In[24]:


train_in = train[0]
train_out = train[1]


# In[32]:


train_in = train_in.reshape((-1,784,1))
train_out = train_out.reshape((-1,1))


# In[56]:


vec_train_out = []
for t in train_out:
    vec_train_out.append(np.array([1 if t[0]==j else 0 for j in range(10)]))
vec_train_out = np.asarray(vec_train_out)


# In[62]:


train_out = vec_train_out
train_out = train_out.reshape((-1,10,1))


# In[ ]:


test_in = test[0]
test_out = test[1]


# In[70]:


vec_test_out = []
for t in test_out:
    vec_test_out.append(np.array([1 if t==j else 0 for j in range(10)]))
vec_test_out = np.asarray(vec_test_out)


# In[71]:


test_out = vec_test_out
test_out = test_out.reshape((-1,10,1))


# In[78]:


data = {'train_input':train_in,'train_output':train_out,'test_input':test_in,'test_output':test_out}


# In[80]:


file = open('mnist_data.pkl','wb')
pickle.dump(data,file)

