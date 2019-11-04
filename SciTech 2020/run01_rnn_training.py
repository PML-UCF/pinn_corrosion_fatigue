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

""" Physics-informed recursive neural network training
"""

import numpy as np
import pandas as pd
import os
import time

from pinn.layers import getScalingDenseLayer

from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint,  TerminateOnNaN
from tensorflow.keras import backend 

import matplotlib.pyplot as plt

from pinn_corrosion_fatigue_bias_model import create_model

from tqdm import tqdm

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
    
    dfcidx = pd.read_csv('rnn_tr_cidx_15000_flights.csv', index_col = None,
                         dtype = myDtype) # loading corrosion index data
    cidx = dfcidx.values[:,1:-1]
    dfdS = pd.read_csv('rnn_tr_dS_15000_flights.csv', index_col = None,
                       dtype = myDtype) # loading mech. load data    
    dS = dfdS.values[:,1:-1]
    dfR = pd.read_csv('rnn_tr_R_15000_flights.csv', index_col = None,
                      dtype = myDtype) # loading stress ratio data
    R = dfR.values[:,1:-1]
    # Filtering infinite crack values
    
    nFleet, nCycles  = np.shape(cidx) 
    
    # RNN inputs
    input_array = np.dstack((dS, R))
    input_array = np.dstack((input_array, cidx))
    
    a0 = pd.read_csv('rnn_initial_crack_length.csv', index_col = None, dtype = myDtype)
    
    a0RNN = np.zeros(input_array.shape[0]) 
    a0RNN[:] = a0.values[:,-1] # initial crack length
    a0RNN = np.reshape(a0RNN,(len(a0RNN),1))
    
    aT = pd.read_csv('rnn_tr_crack_15000_flights.csv', index_col = None, dtype = myDtype)
    
    aTarget = np.zeros(input_array.shape[0]) 
    aTarget[:] = aT.values[:,-1] # initial crack length
    aTarget = np.reshape(aTarget,(len(aTarget),1))
    
    #--------------------------------------------------------------------------
    batch_input_shape = input_array.shape
    #--------------------------------------------------------------------------
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
    
    outputs = aT.values[:,-1]
        
    EPOCHS = 15
    
    arch = np.linspace(0,7,8,dtype = int)
    planes = np.linspace(1,19,19,dtype = int)
    
    cont = 0
    for ii in tqdm(arch):
        aux = planes[cont]
        for jj in aux:
            # Loading MLP info and parameters
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
            Bias_layer.trainable = True
                   
            model = create_model(Bias_layer, low, up, F, beta, gamma, Co, m, a0RNN, batch_input_shape, 
                             selectbias, selectidx, selectdk, selectwk, myDtype)
            
            before = model.predict_on_batch(input_array)
            
            err_b = before[:,0] - outputs
            mse_b = np.mean(err_b**2)
            mepe_b = np.mean(np.sqrt((err_b/outputs)**2))
            
            str_b = "before training\nLoss = %1.1e\nMSE = %1.1e" % (mepe_b, mse_b)
            #--------------------------------------------------------------------------
            checkpoint_path = 'rnn_arch_'+str(ii)+'_15obs_plane_'+str(jj)+'/cp.ckpt'
            checkpoint_dir = os.path.dirname(checkpoint_path)

            # Create checkpoint callback
            cp_callback = ModelCheckpoint(checkpoint_path, save_weights_only=True, save_best_only=True, monitor = 'loss')
            kill = TerminateOnNaN()
                       
            history = model.fit(input_array, aTarget, steps_per_epoch=1, epochs=EPOCHS, verbose=1,
                                callbacks = [cp_callback,kill])
            #--------------------------------------------------------------------------
            results = model.predict_on_batch(input_array)
            
            err_a = results[:,0] - outputs
            mse_a = np.mean(err_a**2)
            mepe_a = np.mean(np.sqrt((err_a/outputs)**2))
                   
            str_a = "after training\nLoss = %1.1e\nMSE = %1.1e" % (mepe_a, mse_a)
            #--------------------------------------------------------------------------    
            # Plot actual x predict 
            fig  = plt.figure()
            fig.clf()
        
            plt.plot(outputs*1e3,outputs*1e3,'--k')
            plt.plot(outputs*1e3,before*1e3,'o', label = str_b)
            plt.plot(outputs*1e3,results*1e3,'o', label = str_a)
        
            plt.title('Training: arch_'+str(ii)+'_plane_'+str(jj))
            plt.xlabel('Actual')
            plt.ylabel('Predicted')
            plt.grid(which = 'both')
            plt.legend(loc='upper left', facecolor='w')
            
            plt.savefig('Plots/training_arch'+str(ii)+'_plane_'+str(jj)+'.png')
            plt.close()
        
        cont+=1
        backend.clear_session()
        
    print('Elapsed time is %s seconds'%(time.time()-start))
    