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
def getScalingDenseLayer(input_location, input_scale, dtype):
    recip_input_scale = np.reciprocal(input_scale)
    
    waux = np.diag(recip_input_scale)
    baux = -input_location*recip_input_scale
    
    dL = Dense(input_location.shape[0], activation = None, input_shape = input_location.shape)
    dL.build(input_shape = input_location.shape)
    dL.set_weights([waux, baux])
    dL.trainable = False
    return dL


def built_model(input_location, input_scale, dtype):
    dLInputScaling = getScalingDenseLayer(input_location, input_scale, dtype)
    model = tf.keras.Sequential([
            dLInputScaling,
            tf.keras.layers.Dense(40),
            tf.keras.layers.Dense(20),
            tf.keras.layers.Dense(10),
            tf.keras.layers.Dense(1, activation = 'softplus'),
            ])

    optimizer = tf.keras.optimizers.RMSprop(0.001)
    model.compile(loss='mean_squared_error',
                  optimizer=optimizer,
                  metrics=['mean_absolute_error', 'mean_squared_error'])
    return model

#--------------------------------------------------------------------------
if __name__ == "__main__":
    myDtype = tf.float32
    
    df = pd.read_csv('corr_MLP_data.csv', index_col = None) # loading corrosion data
    
    inputs = df[['dS','R','Seq','cidx','a']]
    input_location = np.asarray(inputs.min(axis=0))
    input_scale = np.asarray(inputs.max(axis=0)) - np.asarray(inputs.min(axis=0))
    
    outputs = df[['da']]
    outputs_min = np.asarray(outputs.min(axis=0))
    outputs_range = np.asarray(outputs.max(axis=0)) - np.asarray(outputs.min(axis=0))
    outputs_norm = (outputs-outputs_min)/(outputs_range-outputs_min)
    
    model = built_model(input_location, input_scale, myDtype)
    
    model.summary()

    history = model.fit(inputs, outputs_norm, epochs = 25, verbose=1)

    model.save_weights('corr_mlp_weights.h5')
    
    model.save('Corr_MLP.h5')
    
    training_loss = history.history['loss']

    dfval = pd.read_csv('corr_MLP_val.csv')
    inputs_val = dfval[['dS','R','Seq','cidx','a']]
    outputs_val = dfval[['da']]
    outputs_val_min = np.asarray(outputs_val.min(axis=0))
    outputs_val_range = np.asarray(outputs_val.max(axis=0)) - np.asarray(outputs_val.min(axis=0))
    outputs_val_norm = (outputs_val-outputs_val_min)/(outputs_val_range-outputs_val_min)

    results = model.predict(inputs_val)
    #--------------------------------------------------------------------------
    # Create count of the number of epochs
    epoch_count = range(1, len(training_loss) + 1)
    # Visualize loss history
    fig  = plt.figure(1)
    fig.clf()
  
    plt.plot(epoch_count, training_loss, 'r--')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss function (MLP)')
    plt.show();
    #--------------------------------------------------------------------------    
    # Plot delta crack history 
    fig  = plt.figure(2)
    fig.clf()
    
    plt.plot(outputs_val_norm,outputs_val_norm,'-k')
    plt.plot(outputs_val_norm,results,'ob')
    
    plt.title('$\Delta$a corrosion (MLP)')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.grid(which = 'both')
    #--------------------------------------------------------------------------
