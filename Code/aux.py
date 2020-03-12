"""
Auxiliary ESN functions

@author: joshdey
"""
import numpy as np
import math as mt
import ESN as esn
import scipy.io as sio
import matplotlib.pyplot as plt
import ESNteachers

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

def ESN_wrapper(teach_in,teach_out,run,N=400,fb=0,rho=1,W_sig=1,in_sig=.25,fb_sig=1,
                dens=0.25,in_dens=0.8,fb_dens=0.1,a=1,noise=0.0,bias=0,bin_node=0,
                wo=100,MP=True,B=0.1,around=0,order=None,directed=True):
    """
    Wrapper for initializing, training, and running an ESN. Nice for parameter
    searches/generating large data sets for performance comparison. Returns
    ESN.outputs, the generated predictions of the output.
    """
    in_dims=teach_in.shape
    out_dims=teach_out.shape

    ESN=esn.simple_ESN(N=N,K=in_dims[1],L=out_dims[1],a=a,binary_node=bin_node,
                       feedback=fb,W_sig=W_sig,in_sig=in_sig,fb_sig=fb_sig,
                       directed=directed)
    ESN.generate_W(rho=rho,dens=dens)
    ESN.generate_Win(dens=in_dens)
    if fb==1:
        ESN.generate_Wfb(dens=fb_dens)
    ESN.train_ESN(input_dat=teach_in,teacher=teach_out,around=around,order=order,washout=wo,noise=noise,
                  bias=bias,mp=MP,B=B)
    run_dims=run.shape
    ESN.run_ESN(input_dat=run,around=around,order=order,time=run_dims[0],init='last')
    plot=ESN.plot_attractor()

    out=ESN.outputs
    return out

def esn_wrapper(N=400,fb=0,rho=0.7,new_rho=0.7,W_sig=1,in_sig=.25, fb_sig=1,
                dens=0.25,in_dens=0.8,fb_dens=0,a=1,noise=0.0,bias=0,
                bin_node=0,wo=100,MP=False,B=10**-6,around=1,order=1,directed=True,
                teacher_system='Lorenz', directions=0,train_time=400,
                run_time=10000, dt=0.01, init='zero'):
    """
    wrapper for initializing, training, and running the ESN.
    Inputs: ESN parameters, teacher input, teacher output, and running input.
    Output(s): TBD
    """
    if teacher_system=='Rossler':
        xteach, yteach, zteach=ESNteachers.Rossler(l=train_time, dt=dt)
        fullrun=ESNteachers.Rossler(l=run_time,dt=dt)
        run=fullrun[directions]
    else:
        xteach, yteach, zteach=ESNteachers.Lorenz(l=train_time, dt=dt)
        fullrun=ESNteachers.Lorenz(l=run_time, dt=dt)
        run=fullrun[directions]
    if directions==0:
        teach_out = np.concatenate((yteach,zteach), axis=1)
        teach_in=xteach
        act1=fullrun[1]
        act2=fullrun[2]
    elif directions==1:
        teach_out = np.concatenate((xteach,zteach), axis=1)
        teach_in=yteach
        act1=fullrun[0]
        act2=fullrun[2]
    else:
        teach_out = np.concatenate((xteach,yteach), axis=1)
        teach_in=zteach
        act1=fullrun[0]
        act2=fullrun[1]

    in_dims=teach_in.shape
    out_dims=teach_out.shape

    ESN=esn.simple_ESN(N=N, K=in_dims[1], L=out_dims[1], a=a,
                        binary_node=bin_node, feedback=fb, W_sig=W_sig,
                        in_sig=in_sig, fb_sig=fb_sig, directed=directed)
    ESN.generate_W(rho=rho, dens=dens)
    ESN.rescale_W(new_rho=new_rho)
    ESN.generate_Win(dens=in_dens)
    if fb==1:
        ESN.generate_Wfb(dens=fb_dens)
    ESN.train_ESN(input_dat=teach_in,teacher=teach_out,around=around,
                    order=order,washout=wo,noise=noise,bias=bias,mp=MP,B=B)
    run_dims=run.shape
    ESN.run_ESN(input_dat=run,around=around,order=order,time=run_dims[0],
                init='zero')
    out=ESN.outputs
    run1, run2=np.hsplit(out, 2)

    rms=[RMS(run1, act1), RMS(run2, act2)]
    return rms
