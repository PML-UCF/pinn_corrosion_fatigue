# -*- coding: utf-8 -*-
"""
Created on Sat Mar 30 13:31:16 2019

@author: ar679403
"""

# ______          _           _     _ _ _     _   _      
# | ___ \        | |         | |   (_) (_)   | | (_)     
# | |_/ / __ ___ | |__   __ _| |__  _| |_ ___| |_ _  ___ 
# |  __/ '__/ _ \| '_ \ / _` | '_ \| | | / __| __| |/ __|
# | |  | | | (_) | |_) | (_| | |_) | | | \__ \ |_| | (__ 
# \_|  |_|  \___/|_.__/ \__,_|_.__/|_|_|_|___/\__|_|\___|
# ___  ___          _                 _                  
# |  \/  |         | |               (_)                 
# | .  . | ___  ___| |__   __ _ _ __  _  ___ ___         
# | |\/| |/ _ \/ __| '_ \ / _` | '_ \| |/ __/ __|        
# | |  | |  __/ (__| | | | (_| | | | | | (__\__ \        
# \_|  |_/\___|\___|_| |_|\__,_|_| |_|_|\___|___/        
#  _           _                     _                   
# | |         | |                   | |                  
# | |     __ _| |__   ___  _ __ __ _| |_ ___  _ __ _   _ 
# | |    / _` | '_ \ / _ \| '__/ _` | __/ _ \| '__| | | |
# | |___| (_| | |_) | (_) | | | (_| | || (_) | |  | |_| |
# \_____/\__,_|_.__/ \___/|_|  \__,_|\__\___/|_|   \__, |
#                                                   __/ |
#                                                  |___/ 
#														  
# MIT License
# 
# Copyright (c) 2019 Probabilistic Mechanics Laboratory
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# ==============================================================================

""" Corrosion-fatigue MLP
"""
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras.layers import Dense

import matplotlib.pyplot as plt

# =============================================================================
# Function
# =============================================================================
def norm(x):
    x_min = np.min(x, axis = 0)
    x_max = np.max(x, axis = 0)
    x_scale = x_max - x_min
    x_norm = (x - x_min)/x_scale
    return x_norm

def scalar(pred,y):
    y_min = np.min(y, axis = 0)
    y_max = np.max(y, axis = 0)
    y_scale = y_max - y_min
    act = pred*y_scale+y_min
    return act


def built_model():
    model = tf.keras.Sequential([Dense(4, input_shape = [4], trainable = False),
    Dense(60),
    Dense(30),
    Dense(15),
    Dense(5),
    Dense(2, activation = 'softplus')
    ])

    optimizer = tf.keras.optimizers.RMSprop(0.001)

    model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae', 'mse'])
    return model

#--------------------------------------------------------------------------
if __name__ == "__main__":
    myDtype = tf.float32
    
    dfcr = pd.read_csv('Corr_crack.csv', index_col = None) # loading corrosion index data
    cr = dfcr.values[:,1:11]
    cr = cr.flatten()
    auxcr = np.where(cr<=20e-3) 
    cr = cr[auxcr[0][:]] # Removing inf values from dataset
    dfcidx = pd.read_csv('Corr_index.csv', index_col = None) # loading corrosion index data
    cidx = dfcidx.values[:,1:11]
    cidx = cidx.flatten()
    cidx = cidx[auxcr[0][:]]
    dfds = pd.read_csv('Corr_dS.csv', index_col = None) # loading mech. load data
    dS = dfds.values[:,1:11]
    dS = dS.flatten()
    dS = dS[auxcr[0][:]]
    dfR = pd.read_csv('Corr_R.csv', index_col = None) # loading stress ratio data
    R = dfR.values[:,1:11]
    R = R.flatten()
    R = R[auxcr[0][:]]
    dfdai = pd.read_csv('Corr_init.csv', index_col = None) # loading crack initiation data
    dai = dfdai.values[:,1:11]
    dai = dai.flatten()
    dai = dai[auxcr[0][:]]
    dfdap = pd.read_csv('Corr_prop.csv', index_col = None) # loading crack propagation data
    dap = dfdap.values[:,1:11]
    dap = dap.flatten()
    dap = dap[auxcr[0][:]]
    
    input_array = np.column_stack([cr,cidx])
    input_array = np.column_stack([input_array,dS])
    input_array = np.column_stack([input_array,R])
    norm_input = norm(input_array)
 
    
    output_array = np.column_stack([dai,dap])
    norm_output = norm(output_array)

    model = built_model()
    
    model.summary()
    
    history = model.fit(norm_input, norm_output, epochs = 20)
    
# =============================================================================
#     model.save_weights('Corr_MLP_weights.h5')
# =============================================================================
    
    results = model.predict(norm_input, verbose=0, steps=1)
        
    training_loss = history.history['loss']
    #--------------------------------------------------------------------------    
    # Plot delta crack history 
    fig  = plt.figure(1)
    fig.clf()
    
    plt.plot(norm_output[:,0],norm_output[:,0],'-k')
    plt.plot(norm_output[:,0],results[:,0],'ob')
    
    plt.title('Corrosion initiation (MLP)')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.grid(which = 'both')
    #--------------------------------------------------------------------------
    fig  = plt.figure(2)
    fig.clf()
    
    plt.plot(norm_output[:,1],norm_output[:,1],'-k')
    plt.plot(norm_output[:,1],results[:,1],'ob')
    
    plt.title('Corrosion propagation (MLP)')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.grid(which = 'both')

    #--------------------------------------------------------------------------
    # Create count of the number of epochs
    epoch_count = range(1, len(training_loss) + 1)
    # Visualize loss history
    fig  = plt.figure(3)
    fig.clf()
  
    plt.plot(epoch_count, training_loss, 'r--')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss function (MLP)')
    plt.show();
    

    

