"""
Auxiliary ESN functions

@author: joshdey
"""
import numpy as np
import math as mt
import ESN as esn
import scipy.io as sio
import matplotlib.pyplot as plt

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

def ESN_wrapper(teach_in,teach_out,run,N=400,fb=0,rho=1,W_sig=1,in_sig=1,fb_sig=1,
                dens=0.05,in_dens=0.8,fb_dens=0.1,a=1,noise=0.0,bias=1,bin_node=0,
                wo=100,MP=True,B=0.1,around=0,order=None,directed=True):
    """
    Wrapper for initializing, training, and running an ESN. Nice for parameter
    searches/generating large data sets for performance comparison. Returns
    ESN.outputs, the generated predictions of the output.
    """
    in_dims=teach_in.shape
    out_dims=teach_out.shape

    ESN=esn.simple_ESN(N=N,K=in_dims[1],L=out_dims[1],a=a,binary_node=bin_node,
                       feedback=fb,W_sig=W_sig,in_sig=in_sig,fb_sig=fb_sig,directed=directed)
    ESN.generate_W(rho=rho,dens=dens)
    ESN.generate_Win(dens=in_dens)
    if fb==1:
        ESN.generate_Wfb(dens=fb_dens)
    ESN.train_ESN(input_dat=teach_in,teacher=teach_out,around=around,order=order,washout=wo,noise=noise,
                  bias=bias,mp=MP,B=B)
    run_dims=run.shape
    ESN.run_ESN(input_dat=run,around=around,order=order,time=run_dims[0],init='last')

    out=ESN.outputs
    return out

def load_mat(filename):
    mdict=sio.loadmat(filename,appendmat=False)
    arr=mdict['savearr']
    return arr

def load_parms(filename):
    mdict=sio.loadmat(filename,appendmat=False)
    pdict={'N':500,'fb':0,'rho':0.9,'W_sig':1,'in_sig':0.75,'fb_sig':1,
                'dens':0.1,'in_dens':0.9,'fb_dens':0.1,'a':1,'noise':0.0,
                'bias':1,'bin_node':0,'wo':1000,'MP':False,'B':0.1,'around':0,
                'order':None} #Should update to reflect optimized BlockRC parameters
    for key,val in mdict.items():
        pdict[key]=val
    return pdict

def errbars_plot(data,pltnum=1,pltsize=[8,6],s_title=None,titles=None,
                 y_ax='NRMSE',x_ax=None,save=False,path=None):
    dims=data.shape
    x=data[:,0]
    num=int((dims[1]-1)/2)
    fig,axs=plt.subplots(nrows=num,ncols=1,num=pltnum,figsize=pltsize)
    if s_title is not None:
        fig.suptitle(s_title)
    fig.subplots_adjust(hspace=0.5)
    for i in range(0,num):
        ax=axs[i]
        j=2*i+1
        #ax.set_xscale('log')
        ax.errorbar(x,data[:,j],yerr=data[:,j+1],fmt='k.',barsabove=True)
        ax.set_ylabel(y_ax)
        if titles is not None:
            ax.set_title(titles[i])
        if x_ax is not None:
            ax.set_xlabel(x_ax)

    if save is True:
        plt.savefig(path,dpi=300)
