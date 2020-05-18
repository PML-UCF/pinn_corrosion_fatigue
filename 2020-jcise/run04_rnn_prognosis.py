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

""" Physics-informed recurrent neural network prognosis
"""

import numpy as np
import pandas as pd
import time

from pinn.layers import getScalingDenseLayer

from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras import Sequential

import matplotlib.pyplot as plt

from bias_model import create_model
import matplotlib as matplotlib
matplotlib.rc('font', size=14)
myLW = 2.5
myMS = 7

# =============================================================================
# Auxiliary Functions
# =============================================================================
def arch_model(switch, input_location, input_scale):
    dLInputScaling = getScalingDenseLayer(input_location, input_scale)
    if switch == 1:
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
    #--------------------------------------------------------------------------
    # MLP pre- processing
    dfcrl  = pd.read_csv('MLP_crack_tr.csv', index_col = None)
    dfdSl = pd.read_csv('MLP_dS_tr.csv', index_col = None)  
    dfRl = pd.read_csv('MLP_R_tr.csv', index_col = None)   
    dfcidxl = pd.read_csv('MLP_cidx_tr.csv', index_col = None)
    
    dfaux = pd.read_csv('MLP_bias_tr.csv', index_col = None)
    
    F = 2.8 # stress intensity factor
    beta,gamma = -1e8,.68 # Walker model customized sigmoid function parameters
    Co,m = 1.1323e-10,3.859 # Walker model coefficients (similar to Paris law)
    
    selectbias = [0,1,2,3]
    selectidx = [3]
    selectdk = [0,1]
    selectwk = [2]
    
    # best configuration based on run01 and run02 responses 
    # (architecture,auxiliary plane)
    ii,jj = 6,18
    
    a0 = pd.read_csv('prog_initial_crack.csv', index_col = None, dtype = myDtype)
                
    a0RNN = np.zeros(nFleet) 
    a0RNN[:] = a0.values[0:nFleet,-1] # initial crack length
    a0RNN = np.reshape(a0RNN,(len(a0RNN),1))
    
    dfcidx = pd.read_csv('prog_cidx_18000_flights.csv', index_col = None,
                                 dtype = myDtype) # loading corrosion index data
    cidx = dfcidx.values[1:nFleet+1,1:-1]
    dfdS = pd.read_csv('prog_dS_18000_flights.csv', index_col = None,
                       dtype = myDtype) # loading mech. load data    
    dS = dfdS.values[1:nFleet+1,1:-1]
    dfR = pd.read_csv('prog_R_18000_flights.csv', index_col = None,
                      dtype = myDtype) # loading stress ratio data
    R = dfR.values[1:nFleet+1,1:-1]
            
    # RNN inputs
    input_array = np.dstack((dS, R))
    input_array = np.dstack((input_array, cidx))
    batch_input_shape = input_array.shape
           
    aT = pd.read_csv('prog_crack_18000_flights.csv', index_col = None, dtype = myDtype)
    outputs = aT.values[0:nFleet,-1]
    #--------------------------------------------------------------------------
                
    inputs = np.column_stack((dfcrl[str(jj)],dfdSl[str(jj)]))
    inputs = np.column_stack((inputs,dfRl[str(jj)]))
    inputs = np.column_stack((inputs,dfcidxl[str(jj)]))
    
    input_location = np.asarray(inputs.min(axis=0))
    input_scale = np.asarray(inputs.max(axis=0)) - np.asarray(inputs.min(axis=0))
    
    low = np.asarray(dfaux[[str(jj)]].min(axis=0))
    up = np.asarray(dfaux[[str(jj)]].max(axis=0))
    
    Bias_layer = arch_model(ii, input_location, input_scale)
    checkpoint_MLP = 'training_MLP_arch_'+str(ii)+'_plane_'+str(jj)+'/cp.ckpt'
    Bias_layer.load_weights(checkpoint_MLP)
    
    checkpoint_path = 'rnn_arch_'+str(ii)+'_plane_'+str(jj)+'/cp.ckpt'
    #--------------------------------------------------------------------------
    model = create_model(Bias_layer, low, up, F, beta, gamma, Co, m, a0RNN, batch_input_shape, 
                     selectbias, selectidx, selectdk, selectwk, myDtype)
    
    model.load_weights(checkpoint_path)
    pred = model.predict_on_batch(input_array)            
    #--------------------------------------------------------------------------
    # rnn forecast response
    ido = np.argsort(outputs)
    idx = ido - 1 # correction due to csv files index
    
    auxp = pred[idx,0]*1e3
    auxa = outputs[ido]*1e3
    
    idth = np.where(auxp>20)
    idnan = np.where(np.isnan(auxp))
    idp = np.concatenate((idth[0][:],idnan[0][:]))
    
    ida = np.where(auxa>20)
    fp = np.shape(idp)[-1] - np.shape(ida)[-1]
    
    aux_tp = np.where(auxa[idp]>20)
    tp = np.shape(aux_tp)[-1]
    
    
    aux_fn = np.where(auxp[ida]<20)
    fn = np.shape(aux_fn)[-1]
    tn = nFleet-tp-fp-fn
    #--------------------------------------------------------------------------
    fig  = plt.figure()
    fig.clf()

    plt.plot(np.array([0,40]),np.array([0,40]),'--k')
    plt.plot(outputs[ido]*1e3,pred[idx,0]*1e3,'o', color = 'orange', markersize = myMS)
    plt.plot(np.array([20,20]),np.array([0,40]),':k')
    plt.plot(np.array([0,40]),np.array([20,20]),':k')
    plt.text(6,16, 'True negative: \n '+str(tn), ha = 'center', va = 'center')
    plt.text(7,30, 'False positive: \n '+str(fp), ha = 'center', va = 'center')
    plt.text(30,10, 'False negative: \n '+str(fn), ha = 'center', va = 'center')
    plt.text(34,25, 'True positive: \n '+str(tp), ha = 'center', va = 'center')

    plt.xlabel('actual (mm)')
    plt.ylabel('predicted (mm)')
    plt.grid(which = 'both')
    plt.xlim([0,40])
    plt.ylim([0,40])
    plt.tight_layout()
    
    plt.savefig('Plots/prognosis.png')
    #--------------------------------------------------------------------------   

    print('Elapsed time is %s seconds'%(time.time()-start))
    