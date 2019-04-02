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
""" Mechanical propagation sample 
"""
import numpy as np
import pandas as pd
import tensorflow as tf

import matplotlib.pyplot as plt

from tensorflow.python.framework import ops

from model import create_model
from tqdm import tqdm

# =============================================================================
# Main
# =============================================================================
#--------------------------------------------------------------------------
if __name__ == "__main__":
    myDtype = tf.float32  # defining type for the layer
    
    dfcr = pd.read_csv('MLP_Crack_length.csv', index_col = None) # loading corrosion index data
    assets = dfcr.values[:,1:11].shape[-1] # number of considered assets
    cr = dfcr.values[:,1:11]
    cr = cr.flatten()
    auxcr = np.where(cr<=20e-3) 
    cr = cr[auxcr[0][:]] # Removing inf values from dataset
    cycles = int(len(cr)/assets)
    cr = np.reshape(cr,(cycles,assets))
    cr = cr.transpose()
    dfcidx = pd.read_csv('MLP_Corr_index.csv', index_col = None) # loading corrosion index data
    cidx = dfcidx.values[:,1:11]
    cidx = cidx.flatten()
    cidx = cidx[auxcr[0][:]]
    cidx = np.reshape(cidx,(cycles,assets))
    cidx = cidx.transpose()
    dfds = pd.read_csv('MLP_Delta_load.csv', index_col = None) # loading mech. load data
    dS = dfds.values[:,1:11]
    dS = dS.flatten()
    dS = dS[auxcr[0][:]]
    dS = np.reshape(dS,(cycles,assets))
    dS = dS.transpose()
    dfR = pd.read_csv('MLP_Stress_ratio.csv', index_col = None) # loading stress ratio data
    R = dfR.values[:,1:11]
    R = R.flatten()
    R = R[auxcr[0][:]]
    R = np.reshape(R,(cycles,assets))
    R = R.transpose()
    dfSeq = pd.read_csv('MLP_Equiv_stress.csv', index_col = None) # loading crack initiation data
    Seq = dfSeq.values[:,1:11]
    Seq = Seq.flatten()
    Seq = Seq[auxcr[0][:]]
    Seq = np.reshape(Seq,(cycles,assets))
    Seq = Seq.transpose()
    
    nFleet, nCycles  = np.shape(cr) 
    
    # RNN inputs
    input_array = np.dstack((Seq, dS))
    input_array = np.dstack((input_array, R))
    input_array = np.dstack((input_array, cidx))
    inputTensor = ops.convert_to_tensor(input_array, dtype = myDtype)
    
    a0RNN = np.zeros(input_array.shape[0]) 
    a0RNN[:] = cr[:,0] # initial crack length
    a0RNN = ops.convert_to_tensor(a0RNN, dtype=myDtype)
    
    # model parameters
    a,b = -3.73,13.48261 # Sn curve coefficients 
    F = 2.8 # stress intensity factor
    beta,gamma = -1e8,.68 # Walker model customized sigmoid function parameters
    Co,m = 1.1323e-10,3.859 # Walker model coefficients (similar to Paris law)
    alpha,ath = 1e8,.5e-3 # sigmoid selector parameters for initiation-propagation
    eta,cth = 1e8,0 # sigmoid selector parameters for corrosion-mechanical
    #--------------------------------------------------------------------------
    batch_input_shape = input_array.shape
    
    MLP_layer = tf.keras.models.load_model('Corr_MLP.h5')
    MLP_layer.trainable = False
    
    selectsn = [1]
    selectdK = [0,2]
    selectprop = [3]
    selectsig = [0]
    selectcor = [4]
    selectmlp = [0,1,2,3,4]
    
    model = create_model(MLP_layer, a, b, F, beta, gamma, Co, m , alpha, ath, eta, cth, a0RNN, 
                         batch_input_shape, selectsn, selectdK, selectprop, selectsig, selectcor,
                         selectmlp, myDtype, return_sequences = True)
    results = model.predict_on_batch(input_array) # custumized layer prediction
    #--------------------------------------------------------------------------
    fig  = plt.figure(1)
    fig.clf()
    
    plt.plot(1e3*cr[0,:],':k', label = 'asset #1')
    plt.plot(1e3*cr[-1,:],'-g', label = 'asset #10')
    plt.plot(1e3*results[0,:,0],':', label = 'PINN #1')
    plt.plot(1e3*results[-1,:,0],'-', label = 'PINN #10')
    
    
    plt.title('Corr.-Fatigue: complete model')
    plt.xlabel('Cycles')
    plt.ylabel(' crack length [mm]')
    plt.legend(loc=0, facecolor = 'w')
    plt.grid(which = 'both')
    