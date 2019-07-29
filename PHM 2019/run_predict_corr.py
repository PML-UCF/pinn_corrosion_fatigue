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
import tensorflow as tf
import time

import matplotlib.pyplot as plt

from tensorflow.python.framework import ops

from corr_model import create_model

if __name__ == "__main__":
    start = time.time()
    #--------------------------------------------------------------------------
    # pre- processing
    myDtype = tf.float32  # defining type for the layer
    
    dfcidx = pd.read_csv('rnn_predict_idx.csv', index_col = None) # loading corrosion index data
    cidx = dfcidx.values[:,1:-1]
    dfdS = pd.read_csv('rnn_predict_dS.csv', index_col = None) # loading mech. load data    
    dS = dfdS.values[:,1:-1]
    dfR = pd.read_csv('rnn_predict_R.csv', index_col = None) # loading stress ratio data
    R = dfR.values[:,1:-1]
            
    nFleet, nCycles  = np.shape(cidx) 
    
    # RNN inputs
    input_array = np.dstack((dS, R))
    input_array = np.dstack((input_array, cidx))
    inputTensor = ops.convert_to_tensor(input_array, dtype = myDtype)
    
    a0 = pd.read_csv('rnn_predict_a0.csv', index_col = None)
    
    a0RNN = np.zeros(input_array.shape[0]) 
    a0RNN[:] = a0.values[:,-1] # initial crack length
    a0RNN = np.reshape(a0RNN,(len(a0RNN),1))
    a0RNN = ops.convert_to_tensor(a0RNN, dtype=myDtype)
    #--------------------------------------------------------------------------
    batch_input_shape = input_array.shape
    #--------------------------------------------------------------------------
    # loading MLP scaling bounds 
    df = pd.read_csv('MLP_training_data.csv', index_col = None) 
    
    low_C = np.asarray(np.log(df[['C']]).min(axis=0))
    up_C = np.asarray(np.log(df[['C']]).max(axis=0))    
    
    low_m = np.asarray(df[['m']].min(axis=0))
    up_m = np.asarray(df[['m']].max(axis=0))
    
    selectaux = [2,3] # for input selection on the MLPS
    selectdk = [0,1] # for input selection on the StressIntensity layer
    
    F = 2.8   # geometry factor for the stress intensity layer  
    #--------------------------------------------------------------------------    
    # Loading plane MLP     
    MLP_C_p = tf.keras.models.load_model('log_C_MLP_ex.h5')
    MLP_C_p.trainable = False
    
    MLP_m_p = tf.keras.models.load_model('m_MLP_ex.h5')
    MLP_m_p.trainable = False
    
    model = create_model(MLP_C_p, MLP_m_p, low_C, up_C, low_m, up_m, F, a0RNN, 
                         batch_input_shape, selectaux, selectdk, myDtype, return_sequences = True)
    before = model.predict_on_batch(input_array)    
    
    checkpoint_path = "training_20_pts/cp.ckpt"
    model.load_weights(checkpoint_path)
    after = model.predict_on_batch(input_array)  
    #--------------------------------------------------------------------------
    amech = pd.read_csv('rnn_predict_a_mech.csv', index_col = None)
    amech = amech.values[:,-1]
    
    cr = pd.read_csv('rnn_predict_a_value.csv', index_col = None)
    cr = cr.values[:,-1]
    
    idx = np.linspace(0,98,99,dtype = int) 
    idx = np.delete(idx, [20])    # filtering a corrupt datapoint
    
    fig  = plt.figure()
    fig.clf()
    plt.plot(after[idx,-1,0]*1e3,'xk', label = 'rnn prediction')
    plt.plot(amech[idx]*1e3,'sb', label = 'fatigue')
    plt.plot(cr[idx]*1e3,'om', label = 'corrosion-fatigue')
    
    plt.ylabel('a [mm]')
    plt.grid(which = 'both')
    plt.legend(loc=0, facecolor = 'w')
    
    print('Elapsed time is %s seconds'%(time.time()-start))
    