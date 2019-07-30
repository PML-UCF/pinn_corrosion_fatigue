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
""" Corrosion-fatigue MLP for m
"""
import numpy as np
import pandas as pd
import os

from pinn.layers import getScalingDenseLayer

from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint

import matplotlib.pyplot as plt
# =============================================================================
# Function
# =============================================================================
def m_model(input_location, input_scale):
    dLInputScaling = getScalingDenseLayer(input_location, input_scale)
    L1 = Dense(10)
    L2 = Dense(5)
    L3 = Dense(1, activation = 'linear', trainable = True)
    model = Sequential([dLInputScaling,L1,L2,L3], name = 'm_mlp')
    
    optimizer = RMSprop(1e-3)
    model.compile(loss='mse', optimizer=optimizer, metrics=['mae'])
    return model
#--------------------------------------------------------------------------
if __name__ == "__main__":
    myDtype = 'float32'
    
    dft = pd.read_csv('MLP_training_data.csv', index_col = None, dtype = myDtype) # loading training data 
                                                                      # generated from random planes
    inputs = dft[['R','cidx']]
        
    out_prop = dft[['m']]
    #--------------------------------------------------------------------------
    # The following block is related to scaling the inputs of the MLP
    input_location = np.asarray(inputs.min(axis=0))
    input_scale = np.asarray(inputs.max(axis=0)) - np.asarray(inputs.min(axis=0))
    
    out_prop_min = np.asarray(out_prop.min(axis=0))
    out_prop_range = np.asarray(out_prop.max(axis=0)) - np.asarray(out_prop.min(axis=0))
    out_prop_norm = (out_prop-out_prop_min)/(out_prop_range)
    out_prop_location = -out_prop_min*(1/out_prop_range)
    out_prop_scale = 1/out_prop_range
    #--------------------------------------------------------------------------
    # Loading MLP validation data
    dfval = pd.read_csv('MLP_val_data.csv', index_col = None, dtype = myDtype)
    
    inputs_val = dfval[['R','cidx']]
    
    out_prop_val = dfval[['m']]
    #--------------------------------------------------------------------------
    out_prop_val_min = np.asarray(out_prop_val.min(axis=0))
    out_prop_val_range = np.asarray(out_prop_val.max(axis=0)) - np.asarray(out_prop_val.min(axis=0))
    out_prop_val_norm = (out_prop_val-out_prop_val_min)/(out_prop_val_range)
    #--------------------------------------------------------------------------
    checkpoint_path = "training_MLP_m/cp.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)

    # Create checkpoint callback
    cp_callback = ModelCheckpoint(checkpoint_path, save_weights_only=True, verbose=1)
    
    model_prop = m_model(input_location, input_scale)
        
    model_prop.summary()
    
    hist_prop = model_prop.fit(inputs, out_prop_norm, epochs = 10,
                               validation_data = (inputs_val,out_prop_val_norm),
                               callbacks = [cp_callback]) 
    
    training_loss_prop = hist_prop.history['loss']    

    results_prop_norm = model_prop.predict(inputs_val)
    results_prop_norm_location = -out_prop_min*(1/out_prop_range)
    results_prop_norm_scale = 1/out_prop_range
    
    waux_prop = np.reciprocal(results_prop_norm_scale)
    baux_prop = -results_prop_norm_location*np.reciprocal(results_prop_norm_scale)
    results_prop = waux_prop*results_prop_norm+baux_prop
     
    #--------------------------------------------------------------------------
    # Create count of the number of epochs
    epoch_count = range(1, len(training_loss_prop) + 1)
    # Visualize loss history
    fig  = plt.figure(1)
    fig.clf()
  
    plt.plot(epoch_count, training_loss_prop, 'r--')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss function (MLP): m coefficient')
    plt.show();
    #--------------------------------------------------------------------------    
    # Plot actual x predict 
    fig  = plt.figure(2)
    fig.clf()
    
    plt.plot(out_prop_val,out_prop_val,'--k')
    plt.plot(out_prop_val,results_prop,'ob')
    
    plt.title('m (MLP)')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.grid(which = 'both')
    