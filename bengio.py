#!/usr/bin/env python
# coding: utf-8

# In[269]:


import numpy as np
import matplotlib.pyplot as plt
import pickle


# In[270]:


# class Neuron_U:    
#     def __init__(self,T):        
#     def simulate(self,spike_train,I,sigma,epsilon):
#         Vthresh = 1
#         Vspike = 0.5        
#         hidden_layer[0]['soma'] = hidden_layer[0]['soma'] + (-gLK*hidden_layer[0]['soma'])+gB*(hidden_layer[0]['basal'] - hidden_layer[0]['soma']) + gA(hidden_layer[0]['apical'] - hidden_layer[0]['soma']) + (sigma*epsilon)
#         inter[0]['soma'] = inter[0]['soma'] + ((-gLK*inter[0]['soma']) + gD(inter[0]['basal'] - inter[0]['soma']) + I['inter'] + (sigma*epsilon)) 


# In[271]:


# class Neuron_V:    
#     def __init__(self,T):
        
#     def simulate(self,spike_train,I,sigma,epsilon):
#         Vthresh = 1
#         Vspike = 0.5        
        
#         hidden_layer[0]['basal'] = weight_PP[0]*phi(input_layer['soma'])
#         hidden_layer[0]['apical'] = weight_PP[1]*phi(hidden_layer[1]['soma'])+weight_PI[0](inter_layer[0]['soma'])
#         inter_layer[0]['basal'] =  weight_IP[0]*phi(hidden_layer[0]['soma'])
        
#         hidden_layer[1]['basal'] = weight_PP[1]*phi(hidden_layer[1]['soma'])
#         hidden_layer[1]['apical'] = weight_PP[2]*phi(output_layer['soma'])+weight_PI[1](inter_layer[1]['soma'])
#         inter_layer[1]['basal'] =  weight_IP[1]*phi(hidden_layer[1]['soma'])


# In[272]:


# def spiketocurrent(spike_train):
#     current = 0
#     for i in range(len(spike_train)):
#         if(spike_train[i] == 1):
#             current+=1
#         else:
#             current = current*(exp(-1))
#     return current        


# In[273]:


# def reinitialize_spike(spike_trains,num_of_hidden_layers,T):
#     for i in range(784):
#         spikes = [0 for j in range(T)]
#     spike_trains['input'] = spikes
#     for k in range(num_of_hidden_layers):
#         for i in range(500):
#             spikes = [0 for j in range(T)]
#         spike_trains[k] = spikes
#     for i in range(10):
#         spikes = [0 for j in range(T)]
#     spike_trains['output'] = spikes   


# In[274]:


def mnist_to_current(data):
    '''
    Takes input as array or array of arrays
    Maps 0 as current value 2 and 1 as current value 12
    '''
    for i,d in enumerate(data):
        data[i][data[i]>0] = 12
        data[i][data[i]<1] = 2
    return data


# In[287]:


def change_mp(layer,I,epsilon,sigma=0.1,gLK=0.1,gB=1,gA=0.8,gD=1):
    if layer == 0:
        hidden_layer[layer]['basal'] = np.matmul(weights_PP[layer],phi(input_layer['soma']))
    else:
        hidden_layer[layer]['basal'] = np.matmul(weights_PP[layer],phi(hidden_layer[0]['soma']))
    hidden_layer[layer]['soma'] = hidden_layer[layer]['soma'] + (-gLK*hidden_layer[layer]['soma'])+gB*(hidden_layer[layer]['basal'] - hidden_layer[layer]['soma']) + gA*(hidden_layer[layer]['apical'] - hidden_layer[layer]['soma']) + (sigma*epsilon)
    inter_layer[layer]['basal'] = weights_IP[layer]*phi(hidden_layer[layer]['soma'])
#     inter[layer]['soma'] = inter[layer]['soma'] + ((-gLK*inter[layer]['soma']) + gD(inter[layer]['basal'] - inter[layer]['soma']) + I + (sigma*epsilon)) 
    inter_layer[layer]['soma'] = inter_layer[layer]['soma'] + ((-gLK*inter_layer[layer]['soma']) + gD*(inter_layer[layer]['basal'] - inter_layer[layer]['soma']) + I)
    if layer==0:
        hidden_layer[layer]['apical'] = np.matmul(weights_PP[layer+1],phi(hidden_layer[layer+1]['soma']))+np.transpose(np.matmul(np.transpose(weights_PI[layer]),inter_layer[layer]['soma']))
    else:
        hidden_layer[layer]['apical'] = np.matmul(np.transpose(weights_PP[layer+1]),phi(output_layer['basal']))+np.transpose(np.matmul(np.transpose(weights_PI[layer]),inter_layer[layer]['soma']))
    


# In[276]:


def phi(u):
    return 1/(1+np.exp(-u))


# In[277]:


def update_weights(num_of_hidden_layers=2,sigma=0.1,gLK=0.1,gB=1,gA=0.8,gD=1):    
    V_rest = 0
    #layer to layer synaptic weights
    eta_PP = np.zeros((3,1))
    eta_IP = np.zeros((3,1))
    eta_PI = np.zeros((3,1))
    hat_hidden_layer = {}
    for i in range(num_of_hidden_layers):
        hat_hidden_layer[i] = {}
        hat_hidden_layer[i]['basal'] = np.zeros((500,1))
        hat_hidden_layer[i]['soma'] = np.zeros((500,1))
        hat_hidden_layer[i]['apical'] = np.zeros((500,1))
    hat_inter_layer = {}
    for i in range(num_of_hidden_layers):
        hat_inter_layer[i] = {}
        hat_inter_layer[i]['basal'] = np.zeros((500,1))
        hat_inter_layer[i]['soma'] = np.zeros((500,1))
        hat_inter_layer[i]['apical'] = np.zeros((500,1)) 
    hat_output_layer = {}
    hat_output_layer['basal'] = np.zeros((10,1))
    hat_output_layer['soma'] = np.zeros((10,1))    
    eta_PP[0] = 0.01/(0.3**2)
    eta_PP[1] = 0.01/0.3
    eta_PP[2] = 0.01
    hat_hidden_layer[0]['basal'] = (gB/(gLK + gB + gA))*(hidden_layer[0]['basal'])
    weights_PP[0] += eta_PP[0]*np.matmul((phi(hidden_layer[0]['soma']) - phi(hat_hidden_layer[0]['basal'])),np.transpose(input_layer['basal']))
    hat_hidden_layer[1]['basal'] = (gB/(gLK + gB + gA))*(hidden_layer[1]['basal'])
    weights_PP[1] += eta_PP[1]*np.matmul((phi(hidden_layer[1]['soma']) - phi(hat_hidden_layer[1]['basal'])),np.transpose(hidden_layer[0]['basal']))
    hat_output_layer['basal'] = (gB/(gLK + gB + gA))*(output_layer['basal'])
    weights_PP[2] += eta_PP[2]*np.matmul((phi(output_layer['soma']) - phi(hat_output_layer['basal'])),np.transpose(hidden_layer[1]['basal']))
    
    #inter to pyramid synaptic weights
    eta_IP[0] = 2*eta_PP[1]
    eta_IP[1] = 2*eta_PP[2]
    hat_inter_layer[0]['basal'] = (gD/(gLK + gD))*(inter_layer[0]['basal'])
    print(phi(inter_layer[0]['soma']).shape,hat_inter_layer[0]['basal'].shape,hidden_layer[0]['basal'].shape)
    weights_IP[0] += eta_IP[0]*((phi(inter_layer[0]['soma']) - phi(hat_inter_layer[0]['basal']))*(hidden_layer[0]['basal']))
    hat_inter_layer[1]['basal'] = (gD/(gLK + gD))*(inter_layer[1]['basal'])
    weights_IP[1] += eta_IP[1]*((phi(inter_layer[1]['soma']) - phi(hat_inter_layer[1]['basal']))*(hidden_layer[1]['basal']))
    
    #pyramid to inter synaptic weight
    eta_PI[0] = 0.0005
    eta_PI[1] = 0.0005
    weights_PI[0] += eta_PI[0]*((np.full((500,1),V_rest) - hidden_layer[0]['apical'])*(inter_layer[0]['basal']))
    weights_PI[1] += eta_PI[1]*((np.full((500,1),V_rest) - hidden_layer[1]['apical'])*(inter_layer[1]['basal']))


# In[278]:


# def teacher_current(predicted_current,E_inh, E_exc, g_som=0.8):
#     gp_exc = g_som*((predicted_current-E_inh)/(E_exc-E_inh))
#     gp_inh = -g_som*((predicted_current-E_exc)/(E_exc-E_inh))
#     return (gp_exc*(E_exc-predicted_current) + gp_inh*(E_inh-predicted_current))
def teacher_current(predicted_current,target, g_som=0.8):
    return g_som*(target-predicted)


# In[288]:


input_layer = {}
input_layer['basal'] = np.zeros((784,1))
input_layer['soma'] = np.zeros((784,1))

hidden_layer = {}
num_of_hidden_layers = 2
for i in range(num_of_hidden_layers):
    hidden_layer[i] = {}
    hidden_layer[i]['basal'] = np.zeros((500,1))
    hidden_layer[i]['soma'] = np.zeros((500,1))
    hidden_layer[i]['apical'] = np.zeros((500,1))

inter_layer = {}
for i in range(num_of_hidden_layers):
    inter_layer[i] = {}
    inter_layer[i]['basal'] = np.zeros((500,1))
    inter_layer[i]['soma'] = np.zeros((500,1))
    
output_layer = {}
output_layer['basal'] = np.zeros((10,1))
output_layer['soma'] = np.zeros((10,1))

weights_PP = {}
weights_PP[0] = np.random.uniform(-0.1,0.1,392000).reshape(500,784)
weights_PP[1] = np.random.uniform(-0.1,0.1,250000).reshape(500,500)
weights_PP[2] = np.random.uniform(-0.1,0.1,5000).reshape(10,500)

weights_PI = {}
weights_PI[0] = np.random.uniform(-1,1,500).reshape(500,1)
weights_PI[1] = np.random.uniform(-1,1,500).reshape(500,1)

weights_IP = {}
weights_IP[0] = np.random.uniform(-1,1,500).reshape(500,1)
weights_IP[1] = np.random.uniform(-1,1,500).reshape(500,1)


# In[289]:


file = open('binarized_mnist_data.pkl','rb')
data = pickle.load(file)


# In[290]:


train_in,train_out,test_in,test_out = data['train_input'],data['train_output'],data['test_input'],data['test_output']


# In[291]:


train_in = mnist_to_current(train_in)


# In[292]:


input_layer['soma'] = train_in[0]


# In[293]:


change_mp(0,hidden_layer[1]['soma'],1)


# In[295]:


inter_layer[0]['soma'].shape


# In[296]:


# how to include weak nudge?
# change_mp(1,output_layer['soma'],1)


# In[297]:


(output_layer['basal']) = np.transpose(np.matmul(np.transpose(hidden_layer[1]['soma']),np.transpose(weights_PP[2])))


# In[298]:


#what are the values of E_inh E_exc?
predicted = np.asarray([2 if i==0 else 12 for i in train_out[0]])
weak_nudge = teacher_current(predicted,output_layer['basal'])  


# In[299]:


print(weak_nudge)


# In[300]:


#what is V_rest?
update_weights()

