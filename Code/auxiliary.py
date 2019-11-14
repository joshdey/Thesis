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
