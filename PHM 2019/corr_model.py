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
import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Lambda, Multiply, Add

from pinn.layers import CumulativeDamageCell, inputsSelection, StressIntensityRange


def create_model(MLP_C_layer, MLP_m_layer, low_C, up_C, low_m, up_m, F, a0RNN, batch_input_shape, 
                 selectaux, selectdk, myDtype, return_sequences = False, unroll = False):
    
    batch_adjusted_shape = (batch_input_shape[2]+1,) #Adding state
    placeHolder = Input(shape=(batch_input_shape[2]+1,)) #Adding state
    
    filterLayer = inputsSelection(batch_adjusted_shape, selectaux)(placeHolder)
    
    filterdkLayer = inputsSelection(batch_adjusted_shape, selectdk)(placeHolder)
    
    MLP_C_min = low_C
    MLP_C_range = up_C - low_C
    
    MLP_C_layer = MLP_C_layer(filterLayer)
    C_layer = Lambda(lambda x: ((x*MLP_C_range)+MLP_C_min))(MLP_C_layer)
    
    MLP_m_min = low_m
    MLP_m_range = up_m - low_m
    
    MLP_m_layer = MLP_m_layer(filterLayer)
    MLP_scaled_m_layer = Lambda(lambda x: ((x*MLP_m_range)+MLP_m_min))(MLP_m_layer)
    
    dk_input_shape = filterdkLayer.get_shape()
        
    dkLayer = StressIntensityRange(input_shape = dk_input_shape, dtype = myDtype, trainable = False)
    dkLayer.build(input_shape = dk_input_shape)
    dkLayer.set_weights([np.asarray([F], dtype = dkLayer.dtype)])
    dkLayer = dkLayer(filterdkLayer)
    
    ldK_layer = Lambda(lambda x: tf.math.log(x))(dkLayer)
    
    dKm_layer = Multiply()([MLP_scaled_m_layer, ldK_layer])
    
    aux_layer = Add()([C_layer, dKm_layer])
    
    da_layer = Lambda(lambda x: tf.math.exp(x))(aux_layer)

    functionalModel = Model(inputs=[placeHolder], outputs=[da_layer])
    "-------------------------------------------------------------------------"
    CDMCellHybrid = CumulativeDamageCell(model = functionalModel,
                                       batch_input_shape = batch_input_shape,
                                       dtype = myDtype,
                                       initial_damage = a0RNN)
     
    CDMRNNhybrid = tf.keras.layers.RNN(cell = CDMCellHybrid,
                                       return_sequences = return_sequences,
                                       return_state = False,
                                       batch_input_shape = batch_input_shape,
                                       unroll = unroll)
    
    model = tf.keras.Sequential()
    model.add(CDMRNNhybrid)
    model.compile(loss='mse', optimizer=tf.keras.optimizers.RMSprop(learning_rate = 1e-9), metrics=['mae'])
    return model
