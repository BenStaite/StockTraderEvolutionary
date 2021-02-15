# -*- coding: utf-8 -*-
"""
Created on Fri Feb  5 23:51:23 2021

@author: Benst
"""

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import numpy as np
import StockEnv
import time
import pandas
import random

num_batch = 1
num_randoms = 2
num_models = 20
num_inputs = 5
num_actions = 3

wrapper = []
bigVals = []

def createModel():
    model = Sequential()
    model.add(layers.Input(batch_shape = (1,num_batch,num_inputs)))
    model.add(layers.LSTM(50, return_sequences = False, activation="tanh",stateful = True))
   # model.add(layers.LSTM(units = 25, return_sequences = False, activation = 'relu'))
    model.add(layers.Dense(num_actions, activation="softmax"))
    return model

def breed(m1,m2):
    newModel = createModel()
    for j, layer in enumerate(newModel.layers):
        new_weights_for_layer = []
        for k, weight_array in enumerate(layer.get_weights()):
            save_shape = weight_array.shape
            new_weights = weight_array.reshape(-1)
            m1_weights = m1.layers[j].get_weights()[k].reshape(-1)
            m2_weights = m2.layers[j].get_weights()[k].reshape(-1)
            for i, weight in enumerate(new_weights):
                if random.uniform(0, 1) <= 0.5:
                    new_weights[i] = m1_weights[i]
                else:
                    new_weights[i] = m2_weights[i]
            new_weight_array = new_weights.reshape(save_shape)
            new_weights_for_layer.append(new_weight_array)
        newModel.layers[j].set_weights(new_weights_for_layer)
    return newModel

def mutate(model, rate):
    # first itterate through the layers
    newModel = createModel()
    for j, layer in enumerate(model.layers):
        new_weights_for_layer = []
        # each layer has 2 matrizes, one for connection weights and one for biases
        # then itterate though each matrix

        for weight_array in layer.get_weights():
            # save their shape
            save_shape = weight_array.shape
            # reshape them to one dimension 
            one_dim_weight = weight_array.reshape(-1)

            for i, weight in enumerate(one_dim_weight):
                # mutate them like i want
                if random.uniform(0, 1) <= rate:
                    # maybe dont use a complete new weigh, but rather just change it a bit
                    one_dim_weight[i] = random.uniform(0, 2) - 1

            # reshape them back to the original form
            new_weight_array = one_dim_weight.reshape(save_shape)
            # save them to the weight list for the layer
            new_weights_for_layer.append(new_weight_array)

        # set the new weight list for each layer
        newModel.layers[j].set_weights(new_weights_for_layer)
    return newModel

def resetModelStates():
    for i in range(0,num_models):
        for j, layer in enumerate(wrapper[i][0].layers):
            if layer.stateful:
                layer.reset_states()


def createNextGen(t):
    wrapper.sort(key=lambda x: x[1].GetStateValue(x[1].State), reverse=True)
    for i in range(0,num_models):
        print(wrapper[i][1].GetStateValue(wrapper[i][1].State))
    
    print("HighestVal: ", wrapper[0][1].GetStateValue(wrapper[0][1].State))
    bigVals.append(wrapper[0][1].GetStateValue(wrapper[0][1].State))
    for i in range(10,20):
        wrapper[i][0] = mutate(breed(wrapper[i-10][0],wrapper[i-9][0]), 0.001)
        
    for i in range(18,20):
        wrapper[i][0] = createModel()
        
    for i in range(0,num_models):
        wrapper[i][1] = StockEnv.StockEnv("KO", 1000, num_batch,t)
        
    plt.plot(bigVals)
    plt.show()


for i in range(0,num_models):
    wrapper.append([createModel(), StockEnv.StockEnv("KO", 1000, num_batch,0)])
    
template = "Generation: {}, Average Val: {:.3f}, Timestep: {}"

gen = 0
while True:  # Run until solved
    resetModelStates()
    finished = False
    Timestep = 0
    while(not finished):
        tot = 0
        for i in range(0,num_models):
            state = wrapper[i][1].States.to_numpy()
            x = tf.Variable([state], trainable=True, dtype=tf.float32)
            # Predict action probabilities and estimated future rewards
            # from environment state
            action_probs = wrapper[i][0](x)
            # Sample action from action probability distribution
            if(np.isnan(np.min(action_probs))):
                action = 0
                print("NAN")
            else:                
                action = np.random.choice(num_actions, p=np.squeeze(action_probs))
            #print(action)
            # Apply the sampled action in our environment
            wrapper[i][1].Step(action)
            tot += wrapper[i][1].GetStateValue(wrapper[i][1].State)
            
            
        
        if(wrapper[0][1].Finished):
            finished = True
        print(template.format(gen, tot/num_models, Timestep))
        
        if(Timestep % 200 == 0):
            createNextGen(Timestep)
            gen+=1        
        Timestep+=1
        
    createNextGen(Timestep)





