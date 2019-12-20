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

""" Training physics-informed recursive neural network
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

from bias_model import create_model

from tqdm import tqdm

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
    insp = int(15000) # # of flights in the inspection data
    
    dfcidx = pd.read_csv('tr_cidx_'+str(insp)+'_flights.csv', index_col = None,
                         dtype = myDtype) # loading corrosion index data
    cidx = dfcidx.values[:,1:-1]
    dfdS = pd.read_csv('tr_dS_'+str(insp)+'_flights.csv', index_col = None,
                       dtype = myDtype) # loading mech. load data    
    dS = dfdS.values[:,1:-1]
    dfR = pd.read_csv('tr_R_'+str(insp)+'_flights.csv', index_col = None,
                      dtype = myDtype) # loading stress ratio data
    R = dfR.values[:,1:-1]
    # Filtering infinite crack values
    
    nFleet, nCycles  = np.shape(cidx)
    
    a0 = pd.read_csv('initial_crack_length.csv', index_col = None, dtype = myDtype)
    
    aT = pd.read_csv('tr_crack_'+str(insp)+'_flights.csv', index_col = None, dtype = myDtype)
    #--------------------------------------------------------------------------
    # splitting data for x-validation
    auxin = np.linspace(0,nFleet-1,nFleet, dtype = int)
    ridx = np.random.choice(auxin, nFleet, replace=False)
    
    out = 3
    splits = int(np.round(nFleet/out))
    idxout = np.zeros((splits,out), dtype=int)
    idxin = np.zeros((splits,nFleet-out), dtype=int)
    
    for xx in range(splits):
        idxout[xx,:] = ridx[out*xx:out*(xx+1)]
        cont = 0
        for yy in range(nFleet):
            if auxin[yy] not in idxout[xx,:]:
                idxin[xx,cont] = auxin[yy]
                cont+=1
                
    pidx = pd.DataFrame(np.column_stack((idxin,idxout)))
    pidx.to_csv('xval_idx.csv')
    #--------------------------------------------------------------------------
    # RNN scaling info
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
    
    EPOCHS = 5
    
    arch = np.linspace(1,7,7,dtype = int)
    planes = ([1],[16],[1],[16],[16],[18],[5]) 
    
    pred_xval = np.zeros((arch.shape[0],aT.shape[0]))
    split_out = np.zeros(aT.shape[0])
    xabserg = np.zeros((arch.shape[0],aT.shape[0]))
    xerror = np.zeros((arch.shape[0],splits))
    xvalerg = np.zeros(arch.shape[0])
    
    for kk in tqdm(range(splits)):    
        # RNN inputs
        input_array = np.dstack((dS[idxin[kk,:],:], R[idxin[kk,:],:]))
        input_array = np.dstack((input_array, cidx[idxin[kk,:],:]))
        
        a0RNN = np.zeros(input_array.shape[0]) 
        a0RNN[:] = a0.values[idxin[kk,:],-1] # initial crack length
        a0RNN = np.reshape(a0RNN,(len(a0RNN),1))
        
        aTarget = np.zeros(input_array.shape[0]) 
        aTarget[:] = aT.values[idxin[kk,:],-1] # initial crack length
        aTarget = np.reshape(aTarget,(len(aTarget),1)) 
        
        batch_input_shape = input_array.shape
        #--------------------------------------------------------------------------        
        auxval = np.matlib.repmat(idxout[kk,:],1,splits-1)
        idxval = auxval[0]
        
        input_xval = np.dstack((dS[idxval,:], R[idxval,:]))
        input_xval = np.dstack((input_xval, cidx[idxval,:]))
               
        outputs = aT.values[idxout[kk,:],-1]
        
        split_out[kk*out:(kk+1)*out] = aT.values[idxout[kk,:],-1]
        #--------------------------------------------------------------------------
        pcont = 0
        for ii in arch:
            aux = planes[pcont] 
            jj = aux[0]
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
            #--------------------------------------------------------------------------
            checkpoint_xval = 'rnn_arch_'+str(ii)+'_split_'+str(kk)+'/cp.ckpt'
            checkpoint_dir = os.path.dirname(checkpoint_xval) 

            # Create checkpoint callback
            cp_callback = ModelCheckpoint(checkpoint_xval, save_weights_only=True, save_best_only=True, monitor = 'loss')
            kill = TerminateOnNaN() 
                       
            history = model.fit(input_array, aTarget, steps_per_epoch=1, epochs=EPOCHS, verbose=1,
                                callbacks = [cp_callback,kill]) 
            
            pred = model.predict_on_batch(input_xval)
            results = pred[0:out,0]
            pred_xval[pcont,kk*out:(kk+1)*out] = results.copy()
            #--------------------------------------------------------------------------
            err_a = results - outputs 
            xabserg[pcont,kk*out:(kk+1)*out] = err_a.copy()
            
            mse_a = np.mean(err_a**2) 
            
            xerror[pcont,kk] = np.mean(np.sqrt((err_a/outputs)**2)) 
            
            pcont+=1 
            backend.clear_session()
            
    xval_data = pd.DataFrame(np.row_stack((split_out,pred_xval)))
    xval_data.to_csv('xval_pred.csv')
    
    for ee in range(arch.shape[0]):
        xvalerg[ee] = np.sqrt((np.matmul(np.transpose(xerror[ee,:]),xerror[ee,:]))/splits)
            
    
    aplot = np.linspace(np.min(split_out)*1e3,np.max(split_out)*1e3,3)        
        
    fig = plt.figure()
    fig.clf()
    
    plt.plot(aplot,aplot,':k')
    for pp in range(arch.shape[0]):
        plt.plot(split_out*1e3,pred_xval[pp,:]*1e3,'o', label = 'MLP_'+str(pp+1))
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.grid(which = 'both')
    plt.legend(loc='upper left', facecolor='w')
    
    plt.savefig('Plots/xval_pred_comp.png')
    
    idv = np.argsort(split_out)
    
    fig = plt.figure()
    fig.clf()
    
    for pp in range(arch.shape[0]):
        plt.plot(split_out[idv]*1e3,(split_out[idv]-pred_xval[pp,idv])*1e3,'--o', label = 'MLP_'+str(pp+1))
    plt.ylabel('error [mm]')
    plt.xlabel('actual [mm]')
    plt.grid(which = 'both')
    plt.legend(loc='upper left', facecolor='w')
    
    plt.savefig('Plots/xval_pred_error.png')
    
    fig  = plt.figure()
    fig.clf()

    plt.boxplot(np.transpose(100*xerror))
    plt.xlabel('MLP')
    plt.ylabel('% MSE')
    
    fig = plt.figure()
    fig.clf()
    
    for pp in range(arch.shape[0]):
        plt.plot(100*xerror[pp,:],'-o', label = 'MLP_'+str(pp+1))
    plt.ylabel('% MSE')
    plt.grid(which = 'both')
    plt.legend(loc='upper right', facecolor='w')
    
        
    print('Elapsed time is %s seconds'%(time.time()-start))
    