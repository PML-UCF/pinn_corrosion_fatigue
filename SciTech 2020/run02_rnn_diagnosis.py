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

""" Physics-informed recursive neural network diagnosis
"""

import numpy as np
import pandas as pd
import time

from pinn.layers import getScalingDenseLayer

from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras import Sequential

import matplotlib.pyplot as plt

from pinn_corrosion_fatigue_bias_model import create_model
# =============================================================================
# Auxiliary Functions
# =============================================================================
def arch_model(switch, input_location, input_scale):
    dLInputScaling = getScalingDenseLayer(input_location, input_scale)
    if switch == 0:
        L1 = Dense(2, activation = 'tanh')
        L2 = Dense(1, activation = 'linear')
        model = Sequential([dLInputScaling,L1,L2], name = 'bias_mlp')
        model.compile(loss='mse', optimizer=RMSprop(1e-3), metrics=['mae'])
    elif switch == 1:
        L1 = Dense(5, activation = 'tanh')
        L2 = Dense(1, activation = 'linear')
        model = Sequential([dLInputScaling,L1,L2], name = 'bias_mlp')
        model.compile(loss='mse', optimizer=RMSprop(1e-3), metrics=['mae']) 
    elif switch == 2:
        L1 = Dense(10, activation = 'elu')
        L2 = Dense(5, activation = 'elu')
        L3 = Dense(1, activation = 'linear')
        model = Sequential([dLInputScaling,L1,L2,L3], name = 'bias_mlp')
        model.compile(loss='mse', optimizer=RMSprop(1e-3), metrics=['mae'])
    elif switch == 3:
        L1 = Dense(10, activation = 'elu')
        L2 = Dense(5, activation = 'sigmoid')
        L3 = Dense(1, activation = 'elu')
        model = Sequential([dLInputScaling,L1,L2,L3], name = 'bias_mlp')
        model.compile(loss='mse', optimizer=RMSprop(1e-3), metrics=['mae'])
    elif switch == 4:
        L1 = Dense(10, activation = 'tanh')
        L2 = Dense(5, activation = 'tanh')
        L3 = Dense(1, activation = 'elu')
        model = Sequential([dLInputScaling,L1,L2,L3], name = 'bias_mlp')
        model.compile(loss='mse', optimizer=RMSprop(1e-3), metrics=['mae'])
    elif switch == 5:
        L1 = Dense(20, activation = 'tanh')
        L2 = Dense(10, activation = 'elu')
        L3 = Dense(5, activation = 'sigmoid')
        L4 = Dense(1, activation = 'linear', trainable = True)
        model = Sequential([dLInputScaling,L1,L2,L3,L4], name = 'bias_mlp')
        model.compile(loss='mse', optimizer=RMSprop(1e-3), metrics=['mae'])
    elif switch == 6:
        L1 = Dense(20, activation = 'elu')
        L2 = Dense(10, activation = 'sigmoid')
        L3 = Dense(5, activation = 'sigmoid')
        L4 = Dense(1, activation = 'elu', trainable = True)
        model = Sequential([dLInputScaling,L1,L2,L3,L4], name = 'bias_mlp')
        model.compile(loss='mse', optimizer=RMSprop(1e-3), metrics=['mae'])
    elif switch == 7:
        L1 = Dense(40, activation = 'elu')
        L2 = Dense(20, activation = 'sigmoid')
        L3 = Dense(10, activation = 'sigmoid')
        L4 = Dense(1, activation = 'elu', trainable = True)
        model = Sequential([dLInputScaling,L1,L2,L3,L4], name = 'bias_mlp')
        model.compile(loss='mse', optimizer=RMSprop(1e-3), metrics=['mae'])
    return model


if __name__ == "__main__":
    start = time.time()
    #--------------------------------------------------------------------------
    # pre- processing
    myDtype = 'float32'  # defining type for the layer
    nFleet = 150 # integer for filtering the csv files
    
    # MLP pre- processing
    dfcrl  = pd.read_csv('MLP_crack_tr.csv', index_col = None, dtype = myDtype)
    dfdSl = pd.read_csv('MLP_dS_tr.csv', index_col = None, dtype = myDtype)   
    dfRl = pd.read_csv('MLP_R_tr.csv', index_col = None, dtype = myDtype)    
    dfcidxl = pd.read_csv('MLP_cidx_tr.csv', index_col = None, dtype = myDtype)
    
    dfaux= pd.read_csv('MLP_bias_tr.csv', index_col = None, dtype = myDtype)
    
    F = 2.8 # stress intensity factor
    beta,gamma = -1e8,.68 # Walker model customized sigmoid function parameters
    Co,m = 1.1323e-10,3.859 # Walker model coefficients (similar to Paris law)
    
    selectbias = [0,1,2,3]
    selectidx = [3]
    selectdk = [0,1]
    selectwk = [2]
    
    arch = np.array([3]) # best architecture based on run01 response
    planes = np.array([2]) # best initial guess based on run01 response
    
    dfcidx = pd.read_csv('diag_cidx_15000_flights.csv', index_col = None,
                                     dtype = myDtype) # loading corrosion index data
    cidx = dfcidx.values[1:nFleet+1,1:-1]
    dfdS = pd.read_csv('diag_dS_15000_flights.csv', index_col = None,
                       dtype = myDtype) # loading mech. load data    
    dS = dfdS.values[1:nFleet+1,1:-1]
    dfR = pd.read_csv('diag_R_15000_flights.csv', index_col = None,
                      dtype = myDtype) # loading stress ratio data
    R = dfR.values[1:nFleet+1,1:-1]
            
    # RNN inputs
    input_array = np.dstack((dS, R))
    input_array = np.dstack((input_array, cidx))
    
    batch_input_shape = input_array.shape
    
    aT = pd.read_csv('diag_crack_15000_flights.csv', index_col = None, dtype = myDtype)
    outputs = aT.values[0:nFleet,-1]
        
    a0 = pd.read_csv('prog_initial_crack.csv', index_col = None, dtype = myDtype)
                
    a0RNN = np.zeros(nFleet) 
    a0RNN[:] = a0.values[0:nFleet,-1] # initial crack length
    a0RNN = np.reshape(a0RNN,(len(a0RNN),1))
    
    aM = pd.read_csv('diag_crack_mech_15000_flights.csv', index_col = None, dtype = myDtype)
    mech = aM.values[0:nFleet,-1]    
        
    inputs = np.column_stack((dfcrl[str(planes[0])],dfdSl[str(planes[0])]))
    inputs = np.column_stack((inputs,dfRl[str(planes[0])]))
    inputs = np.column_stack((inputs,dfcidxl[str(planes[0])]))
    
    input_location = np.asarray(inputs.min(axis=0))
    input_scale = np.asarray(inputs.max(axis=0)) - np.asarray(inputs.min(axis=0))
    
    low = np.asarray(dfaux[[str(planes[0])]].min(axis=0))
    up = np.asarray(dfaux[[str(planes[0])]].max(axis=0))
    #--------------------------------------------------------------------------
    Bias_layer = arch_model(arch[0], input_location, input_scale)
    checkpoint_MLP = 'training_MLP_arch_'+str(arch[0])+'_plane_'+str(planes[0])+'/cp.ckpt'
    Bias_layer.load_weights(checkpoint_MLP)
       
    model = create_model(Bias_layer, low, up, F, beta, gamma, Co, m, a0RNN, batch_input_shape, 
                     selectbias, selectidx, selectdk, selectwk, myDtype)
    
    checkpoint_path = 'rnn_arch_'+str(arch[0])+'_15obs_plane_'+str(planes[0])+'/cp.ckpt'
    model.load_weights(checkpoint_path)
    
    pred = model.predict_on_batch(input_array)
    
    ido = np.argsort(outputs)
    idx = ido - 1 # correction due to csv files index
    
    fig  = plt.figure()
    fig.clf()

    plt.plot(outputs[ido]*1e3,'om', label = 'corrosion-fatigue')
    plt.plot(pred[idx,0]*1e3,'xk', label = 'rnn prediction')
    plt.plot(mech[ido]*1e3,'sb',label = 'fatigue')
    
    plt.ylabel('a[mm]')
    plt.legend(loc=0, facecolor = 'w')
    plt.grid(which = 'both') 
    plt.ylim([0,20])  
    plt.savefig('Plots/diagnosis_arch'+str(arch[0])+'_15obs_plane_'+str(planes[0])+'.png')
                    
    print('Elapsed time is %s seconds'%(time.time()-start))
    