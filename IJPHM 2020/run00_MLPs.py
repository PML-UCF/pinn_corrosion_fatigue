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
""" Corrosion-fatigue data generator (MLP)
"""
import numpy as np
import pandas as pd
import os
import time

from pinn.layers import getScalingDenseLayer

from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint, TerminateOnNaN

import matplotlib.pyplot as plt
from tqdm import tqdm
# =============================================================================
# Functions
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
    myDtype = 'float32'
    
    arch = np.linspace(1,7,7,dtype = int)
    planes = np.linspace(1,19,19,dtype = int)  
    
    for ii in tqdm(arch):
        for jj in planes:        
            dfcr = pd.read_csv('MLP_crack_tr.csv', index_col = None)
            cr = dfcr[str(jj)]
            dfdS = pd.read_csv('MLP_dS_tr.csv', index_col = None)
            dS = dfdS[str(jj)]
            dfR = pd.read_csv('MLP_R_tr.csv', index_col = None)
            R = dfR[str(jj)]
            dfcidx = pd.read_csv('MLP_cidx_tr.csv', index_col = None)
            cidx = dfcidx[str(jj)]
            
            dfout = pd.read_csv('MLP_bias_tr.csv', index_col = None)
            out_bias = dfout[[str(jj)]] 
            
            inputs = np.column_stack((cr,dS))
            inputs = np.column_stack((inputs,R))
            inputs = np.column_stack((inputs,cidx))
            
            input_location = np.asarray(inputs.min(axis=0))
            input_scale = np.asarray(inputs.max(axis=0)) - np.asarray(inputs.min(axis=0))
            
            dfcrv = pd.read_csv('MLP_crack_val.csv', index_col = None)
            crv = dfcrv[str(jj)]
            dfdSv = pd.read_csv('MLP_dS_val.csv', index_col = None)
            dSv = dfdSv[str(jj)]
            dfRv = pd.read_csv('MLP_R_val.csv', index_col = None)
            Rv = dfRv[str(jj)]
            dfcidxv = pd.read_csv('MLP_cidx_val.csv', index_col = None)
            cidxv = dfcidxv[str(jj)]
            
            dfoutv = pd.read_csv('MLP_bias_val.csv', index_col = None)
            out_bias_val = dfoutv[[str(jj)]] 
            
            inputs_val = np.column_stack((crv,dSv))
            inputs_val = np.column_stack((inputs_val,Rv))
            inputs_val = np.column_stack((inputs_val,cidxv))
                
            out_bias_min = np.asarray(out_bias.min(axis=0))
            out_bias_range = np.asarray(out_bias.max(axis=0)) - np.asarray(out_bias.min(axis=0))
            out_bias_norm = (out_bias-out_bias_min)/(out_bias_range)
            out_bias_location = -out_bias_min*(1/out_bias_range)
            out_bias_scale = 1/out_bias_range
                 
            out_bias_val_min = np.asarray(out_bias_val.min(axis=0))
            out_bias_val_range = np.asarray(out_bias_val.max(axis=0)) - np.asarray(out_bias_val.min(axis=0))
            out_bias_val_norm = (out_bias_val-out_bias_val_min)/(out_bias_val_range)
            
            checkpoint_path = 'training_MLP_arch_'+str(ii)+'_plane_'+str(jj)+'/cp.ckpt'
            checkpoint_dir = os.path.dirname(checkpoint_path)
    
            # Create checkpoint callback
            cp_callback = ModelCheckpoint(checkpoint_path, save_weights_only=True)
            kill = TerminateOnNaN()
            
            model_bias = arch_model(ii, input_location, input_scale)
        
            hist_bias = model_bias.fit(inputs, out_bias_norm, epochs = 10, verbose = 0, 
                            validation_data = (inputs_val,out_bias_val_norm),
                            callbacks = [cp_callback,kill]) 
        
            results_bias_norm = model_bias.predict(inputs_val)
            results_bias_norm_location = -out_bias_min*(1/out_bias_range)
            results_bias_norm_scale = 1/out_bias_range   
            #--------------------------------------------------------------------------
            # Plot actual x predict 
        
            fig  = plt.figure()
            fig.clf()
        
            plt.plot(out_bias_val_norm,out_bias_val_norm,'--k')
            plt.plot(out_bias_val_norm,results_bias_norm,'ob')
        
            plt.title('bias (MLP)')
            plt.xlabel('Actual')
            plt.ylabel('Predicted')
            plt.grid(which = 'both')