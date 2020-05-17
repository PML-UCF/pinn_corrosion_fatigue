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

""" Prediction physics-informed recursive neural network
"""
import numpy as np
import pandas as pd
import time

import matplotlib.pyplot as plt

from pinn.layers import getScalingDenseLayer

from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras import Sequential

from corrosion_fatigue_model import create_model

# =============================================================================
# Function
# =============================================================================
def logC_model(input_location, input_scale):
    dLInputScaling = getScalingDenseLayer(input_location, input_scale)
    L1 = Dense(40, activation = 'elu')
    L2 = Dense(20, activation = 'elu')
    L3 = Dense(10, activation = 'elu')
    L4 = Dense(5, activation = 'elu')
    L5 = Dense(1, activation = 'elu', trainable = False)
    model = Sequential([dLInputScaling,L1,L2,L3,L4,L5], name = 'c_mlp')
    
    optimizer = RMSprop(1e-3)
    model.compile(loss='mse', optimizer=optimizer, metrics=['mae'])
    return model

def m_model(input_location, input_scale):
    dLInputScaling = getScalingDenseLayer(input_location, input_scale)
    L1 = Dense(10)
    L2 = Dense(5)
    L3 = Dense(1, activation = 'linear', trainable = False)
    model = Sequential([dLInputScaling,L1,L2,L3], name = 'm_mlp')
    
    optimizer = RMSprop(1e-3)
    model.compile(loss='mse', optimizer=optimizer, metrics=['mae', 'accuracy'])
    return model

if __name__ == "__main__":
    start = time.time()
    #--------------------------------------------------------------------------
    # pre- processing
    myDtype = 'float32'  # defining type for the layer
    
    dfcidx = pd.read_csv('rnn_predict_7yr_cidx.csv', index_col = None, dtype = myDtype) # loading corrosion index data
    cidx = dfcidx.values[:,1:-1]
    dfdS = pd.read_csv('rnn_predict_7yr_dS.csv', index_col = None, dtype = myDtype) # loading mech. load data    
    dS = dfdS.values[:,1:-1]
    dfR = pd.read_csv('rnn_predict_7yr_R.csv', index_col = None, dtype = myDtype) # loading stress ratio data
    R = dfR.values[:,1:-1]
            
    nFleet, nCycles  = np.shape(cidx) 
    
    # RNN inputs
    input_array = np.dstack((dS, R))
    input_array = np.dstack((input_array, cidx))
    
    a0 = pd.read_csv('rnn_predict_initial_crack.csv', index_col = None, dtype = myDtype)
    
    a0RNN = np.zeros(input_array.shape[0]) 
    a0RNN[:] = a0.values[:,-1] # initial crack length
    a0RNN = np.reshape(a0RNN,(len(a0RNN),1))
    #--------------------------------------------------------------------------
    batch_input_shape = input_array.shape
    #--------------------------------------------------------------------------
    # Loading MLP info and parameters
    dfmlp = pd.read_csv('MLP_training_data.csv', index_col = None, dtype = myDtype) # loading training data 
                                                                      # generated from random planes
    inputs = dfmlp[['R','cidx']]
    
    input_location = np.asarray(inputs.min(axis=0))
    input_scale = np.asarray(inputs.max(axis=0)) - np.asarray(inputs.min(axis=0))
    
    MLP_C_layer = logC_model(input_location, input_scale)
    checkpoint_logC = "training_MLP_logC/cp.ckpt"
    MLP_C_layer.load_weights(checkpoint_logC)
    
    MLP_m_layer = m_model(input_location, input_scale)
    checkpoint_m = "training_MLP_m/cp.ckpt"
    MLP_m_layer.load_weights(checkpoint_m)
        
    low_C = np.asarray(np.log10(dfmlp[['C']]).min(axis=0))
    up_C = np.asarray(np.log10(dfmlp[['C']]).max(axis=0))    
    
    low_m = np.asarray(dfmlp[['m']].min(axis=0))
    up_m = np.asarray(dfmlp[['m']].max(axis=0)) 
    
    selectaux = [2,3] # for input selection on the MLPS
    selectdk = [0,1] # for input selection on the StressIntensity layer
    
    F = 2.8   # geometry factor for the stress intensity layer  
    #--------------------------------------------------------------------------    
    model = create_model(MLP_C_layer, MLP_m_layer, low_C, up_C, low_m, up_m, F, a0RNN, 
                         batch_input_shape, selectaux, selectdk, myDtype, return_sequences = True)
    
    checkpoint_path = "training_20_pts/cp.ckpt"
    model.load_weights(checkpoint_path)
    prediction = model.predict_on_batch(input_array)  
    #--------------------------------------------------------------------------
    mech = pd.read_csv('rnn_predict_cracks_pure_fatigue.csv', index_col = None, dtype = myDtype)
    mech = mech.values[:,-1]
    
    cr = pd.read_csv('rnn_predict_7yr_crack_length.csv', index_col = None, dtype = myDtype)
    cr = cr.values[:,-1]
    
    idx = np.linspace(0,98,99,dtype = int) 
    idx = np.delete(idx, [20])    # filtering an outlier
    
    fig  = plt.figure()
    fig.clf()
    plt.plot(prediction[idx,-1,0]*1e3,'xk', label = 'rnn prediction')
    plt.plot(mech[idx]*1e3,'sb', label = 'fatigue')
    plt.plot(cr[idx]*1e3,'om', label = 'corrosion-fatigue')
    
    plt.ylabel('a [mm]')
    plt.grid(which = 'both')
    plt.legend(loc=0, facecolor = 'w')
    
    print('Elapsed time is %s seconds'%(time.time()-start))
    