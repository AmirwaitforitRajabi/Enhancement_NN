#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 13:12:14 2020

@author: yasong
"""

from scipy.io import wavfile
import numpy as np
import os
import matplotlib.pyplot as plt

"""
------------------------------------------------------------------------
active speech level based on ITU p.56 (https://www.itu.int/rec/T-REC-P.56/en)
Parameters:
    x : 1-D array of float between (-1, 1), OR
        1-D array of int16
    fs: sample rate
Returns:
    asl            : active speech level in [dB]
    long_term_level: RMS in [dB]
------------------------------------------------------------------------
"""
def interp(a0 , a1, c0, c1, idx, M):
    
    ka = a1 - a0
    kc = c1 - c0
    ba = a1 - ka * idx
    bc = c1 - kc * idx
    
    j0 = (M + bc - ba) / (ka - kc)

    asl = ka * j0 + ba
    c_thr = kc * j0 + bc
    
    return asl, c_thr, j0

def actlev(x, fs):
    
    if isinstance(x[0], np.int16):
        x = x / 2**15
    
    if max(abs(x))>1:
        raise ValueError('Normalize signal to (-1, 1) first.')
    
    # parameters
    EPS = np.finfo("float").eps
    T = 0.03                # time constant of smoothing in seconds
    g = np.exp(-1/(T*fs))   # coefficient of smoothing
    H = 0.20                # Hangover time in seconds
    I = int(np.ceil(H*fs))  
    M = 15.9                # Margin between c_dB and a_dB
    nbits = 16

    a = np.zeros(nbits-1)                       # activity count
    c = 0.5**np.arange(nbits-1, 0, step=-1)     # threshold level

    # initialize
    h = np.ones(nbits)*I                        # Hangover count
    sq = 0
    p = 0
    q = 0
    asl = -100

    sq = sum(x**2)
    c_dB = 20*np.log10(c)
    lond_term_level = 10*np.log10(np.mean(x**2) + EPS)

    for xi in x:
        p = g * p + (1-g) * abs(xi)
        q = g * q + (1-g) * p

        for j in range(nbits-1):
            if q >= c[j]:
                a[j] += 1
                h[j] = 0
            elif h[j] < I:
                a[j] += 1;
                h[j] += 1

    a = np.divide(sq, a, out=np.ones_like(a)*(10**(-10)), where=a!=0)
    a_dB = 10*np.log10(a)
            
    delta = a_dB - c_dB - M 
    idx = np.where(delta <= 0)[0]
    if len(idx)>0:
        idx = idx[0]
        if idx>0:
            asl,c_thr,j0 = interp(a_dB[idx-1], a_dB[idx], c_dB[idx-1], c_dB[idx], idx, M)
        else:
            asl = a_dB[idx]

    return asl, lond_term_level

"""
------------------------------------------------------------------------
return the list of .wav files in the folder
usage:
    fileList = get_wav_file(folder, [])
Parameters:
    path     : folder path
    wavList  : file list to extend
Returns:
    wavList  : .wav file list
------------------------------------------------------------------------
"""
def get_wav_file(path, wavList):
    
    root, subfolders, files = list(os.walk(path))[0]
    for folder in subfolders:
        wavList = get_wav_file(os.path.join(root, folder), wavList)
        
    for f in files:
        if f.endswith('.wav'):
            wavList.append(os.path.join(root, f))
    
    return wavList

"""
------------------------------------------------------------------------
read .wav file (int16)
Parameters:
    file : string, output name
    unit : optional boolean, if True, int16 signal will be scaled into (-1,1)
    l    : optional integer, length of returned signal
Returns:
    fs : sample rate
    x  : signal
------------------------------------------------------------------------
"""
def readwav(file, unit=True, l=None):
    
    [fs, x] = wavfile.read(file, True)
    
    if unit:
        x0 = x.reshape(-1)[0]
        if isinstance(x0, np.int16):
            x = x / 2**15
        if isinstance(x0, np.int32):
            x = x / 2**31
    
    if l:
        x = x[0:l]
    
    return fs, x