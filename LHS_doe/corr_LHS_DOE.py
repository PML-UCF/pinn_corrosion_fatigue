# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 15:12:48 2019

@author: ar679403
"""

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

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd
import time

# =============================================================================
# import sys
# sys.path.append('../../')
# from pmlDOE.src import pmlDOE as pmlDOE
# =============================================================================

import pmlDOE
# =============================================================================
# Functions
# =============================================================================
def chen(ao,Cidex):
    beta = .29
    Cp = Cidex*2.092258600806943e-19 # or median 5.574291205373569e-20
    B = Cp/(2*np.pi*beta)
    da = B*(ao**-2)
    return da

def walker(dS,R,cidx,a):
    m = 1.853
    Co = 2.2415e-8 
    Kt = 2.8
    if R<0:
        gamma = 0
        C = Co/((1-R)**(m*(1-gamma)))
    else:
        gamma = .68
        C = Co/((1-R)**(m*(1-gamma)))
    dk = Kt*dS*np.sqrt(np.pi*a)
    da = C*(dk**m)
    return da

def daestimator(data,ath):
    dS = data[:,0]
    R = data[:,1]
    cidx = data[:,3]
    a = data[:,-1]
    
    dai = chen(a,cidx)
    dap = walker(dS,R,cidx,a)
    if a < ath:
        da = dai
    else:
        da = dap
    return da,dai,dap


if __name__ == "__main__":
    
    start = time.time()
    npnts = 5000
    Xolhs = pmlDOE.lhs(nvar = 5, npnts = npnts, iterations = 10)
    lowerBounds = np.asarray([[0,-.23,0,0,0]])
    upperBounds = np.asarray([[84,.64,1.86,1,.02]])
    
    n, m = np.shape(Xolhs)

    fromDomain = np.concatenate((np.zeros((1,m)), np.ones((1,m))), axis = 0)
    toDomain   = np.asarray([lowerBounds, upperBounds])
    data      = pmlDOE.mapArray(Xolhs, fromDomain, toDomain)
    
    scaledXolhs =  np.repeat(lowerBounds, npnts, axis = 0) + Xolhs * (upperBounds - lowerBounds)

    ath = .5e-3
    
    da, dai, dap = daestimator(data,ath)
    
    info_array = np.column_stack([data,da])
    info_array = np.column_stack([info_array,dai])
    info_array = np.column_stack([info_array,dap])
        
    
    df = pd.DataFrame(info_array, columns = ['dS', 'R', 'Seq', 'cidx', 'a','da','dai','dap'])
    
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(xs = df['a'], ys = df['dS'], zs = df['da'])
    
    df.to_csv('corr_MLP_data.csv', index = False)
    
    print('Elapsed time is %s seconds'%(time.time()-start))