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

from tensorflow import sign, cond
from tensorflow.keras import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Lambda, Concatenate, Add, Multiply, RNN
from tensorflow.keras.optimizers import RMSprop
from tensorflow.math import is_inf

from keras import backend as K

from pinn.layers import inputsSelection, CumulativeDamageCell,StressIntensityRange, WalkerModel

def crmse(y_true, y_pred):
    aux = K.sqrt(K.mean(K.square(y_pred - y_true)))
    loss = cond(is_inf(aux), lambda: 1., lambda: aux)
    return loss

def mepe(y_true, y_pred):
    loss = K.mean(K.sqrt(K.square((y_pred - y_true)/y_true)))
    return loss

def mape(y_true, y_pred):
    loss = K.sqrt(K.square(K.max((y_pred - y_true)/y_true)))
    return loss

def perc_err(y_true, y_pred):
    loss = K.sqrt(K.square(K.mean((y_pred - y_true)/y_true)))
    return loss


def create_model(Bias_layer, low, up, F, beta, gamma, Co, m, a0RNN, batch_input_shape, selectbias, 
                 selectidx, selectdk, selectwk, myDtype, return_sequences = False, unroll = False):
    
    batch_adjusted_shape = (batch_input_shape[2]+1,) #Adding state
    placeHolder = Input(shape=(batch_input_shape[2]+1,)) #Adding state
    
    filterBias = inputsSelection(batch_adjusted_shape, selectbias)(placeHolder)
    
    filterSig = inputsSelection(batch_adjusted_shape, selectidx)(placeHolder)
    
    filterdK = inputsSelection(batch_adjusted_shape, selectdk)(placeHolder)
    
    filterda = inputsSelection(batch_adjusted_shape, selectwk)(placeHolder)
    
    MLP_min = low
    MLP_range = up - low
    
    Bias_layer = Bias_layer(filterBias)
    MLP = Lambda(lambda x: ((x*MLP_range)+MLP_min))(Bias_layer)
    
    Filter = Lambda(lambda x: sign(x))(filterSig)
    
    Bias_filtered_layer = Multiply()([MLP, Filter])
    
    dk_input_shape = filterdK.get_shape()
    
    dkLayer = StressIntensityRange(input_shape = dk_input_shape, dtype = myDtype, trainable = False)
    dkLayer.build(input_shape = dk_input_shape)
    dkLayer.set_weights([np.asarray([F], dtype = dkLayer.dtype)])
    dkLayer = dkLayer(filterdK)
    
    wmInput = Concatenate(axis = -1)([dkLayer, filterda])
    wm_input_shape = wmInput.get_shape()
    
    wmLayer = WalkerModel(input_shape = wm_input_shape, dtype = myDtype, trainable = False)
    wmLayer.build(input_shape = wm_input_shape)
    wmLayer.set_weights([np.asarray([beta, gamma, Co, m], dtype = wmLayer.dtype)])
    wmLayer = wmLayer(wmInput)
    
    da_layer = Add()([Bias_filtered_layer, wmLayer])

    functionalModel = Model(inputs=[placeHolder], outputs=[da_layer])
    "-------------------------------------------------------------------------"
    CDMCellHybrid = CumulativeDamageCell(model = functionalModel,
                                       batch_input_shape = batch_input_shape,
                                       dtype = myDtype,
                                       initial_damage = a0RNN)
     
    CDMRNNhybrid = RNN(cell = CDMCellHybrid, return_sequences = return_sequences, 
                       return_state = False, batch_input_shape = batch_input_shape, unroll = unroll)
    
    model = Sequential()
    model.add(CDMRNNhybrid)
    model.compile(loss=mepe, optimizer=RMSprop(1e-11), metrics=['mse'])
    return model
