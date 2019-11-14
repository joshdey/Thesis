"""
Auxiliary ESN functions

@author: joshdey
"""
import numpy as np
import math as mt
import scipy.signal as sps
import ESN as esn
import scipy.stats as scs
import scipy.io as sio
import os
import re
import matplotlib.pyplot as plt
import zlib as zl
import copy
from multiprocessing import Pool,cpu_count

def RMS(act,exp):
    """
    act: actual generated values
    exp: expected values

    Both inputs should be numpy arrays
    """
    act=act.reshape((act.shape[0],1))
    exp=exp.reshape((exp.shape[0],1))
    try:
        resid=np.square(act-exp)
    except:
        raise IndexError("Indices of act and exp do not match")
    resid=np.sum(resid)
    n=act.shape[0]
    rms=mt.sqrt(resid/n)

    return rms

def RMS_cml(act,exp):
    """
    act:(time,vals) array of generated values
    exp:(time,vals) array of expected values
    """
    resid=np.square(act-exp)
    resid=np.sum(resid,axis=1)
    #n=act.shape[1]
    resid=resid/act.shape[1]
    resid=np.sum(resid)/act.shape[0]
    rms=mt.sqrt(resid)

    return rms

def RMS_over_t(act,exp):
    """
    Returns the RMS at each time step for many time steps

    act:(time,vals) array of generated values
    exp:(time,vals) array of expected values

    returns: (time, RMS) array of rms values
    """
    try:
        resid=np.square(act-exp)
    except:
        raise IndexError("Indices of act and exp probably do not match")
    resid=np.sum(resid,axis=1)
    n=act.shape[1]
    rms=np.sqrt(resid/n)

    return rms

def fir_downsample(data,new_dt,old_dt):
    """
    Downsamples data for use in ESNs. Useful if teacher data is sampled at a
    small dt, but less accuracy is wanted for feeding to the ESN.

    data should be a 1D numpy array.
    """
    resamp=int(new_dt/old_dt)
    new_data=sps.decimate(data,resamp,ftype='fir',axis=0)
    return new_data

def fft_downsample(data,new_dt,old_dt):
    old_len=data.shape[0]
    resamp=new_dt/old_dt
    new_len=int(old_len/resamp)
    new_data=sps.resample(data,new_len)
    return new_data

def taylor_exp(x_in,coeffs):
    dim=coeffs.shape[0]
    x_out=0
    for i in range(0,dim):
        x_out+=coeffs[i]*np.power(x_in,i)
    return x_out

def ReLU(x_in,coeffs):
    scaled=coeffs*x_in
    x_out=np.maximum(scaled,0)
    return x_out

def autocorrelation(x,norm=False):
    l=x.shape[0]
    x=x-np.mean(x)
    var=np.var(x)
    try:
        corr=np.correlate(x,x,mode='full')[-l:]
    except:
        raise IndexError('Array was possibly not 1D')
    #corr=corr[int(corr.size/2):]
    res=corr/(var*(np.arange(l,0,-1)))
    return res
