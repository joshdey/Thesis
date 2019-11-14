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

def estimated_autocorrelation(x):
    """
    http://stackoverflow.com/q/14297012/190597
    http://en.wikipedia.org/wiki/Autocorrelation#Estimation
    """
    n = len(x)
    variance = x.var()
    x = x-x.mean()
    r = np.correlate(x, x, mode = 'full')[-n:]
    assert np.allclose(r, np.array([(x[:n-k]*x[-(n-k):]).sum() for k in range(n)]))
    result = r/(variance*(np.arange(n, 0, -1)))
    return result

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

def NaiveCommittee(memnum,train_in,train_out,run,N=500,fb=0,rho=1,W_sig=1,in_sig=1,fb_sig=1,
                dens=0.05,in_dens=0.8,fb_dens=0.1,a=1,noise=0.0,bias=1,bin_node=0,
                wo=100,MP=True,B=0.1,around=0,order=None):
    """
    memnum: number of committee members.
    train_in: list of training input data arrays. Each list entry should be the array of
        data to be passed to a given committee member.
    train_out: list of teacher data arrays. Each list entry should be the array of
        data to be passed to a given committee member.
    run: list of input data arrays. Each list entry should be the array of
        data to be passed to a given committee member.

    Will only have N, rho, fb as numbers or lists. Add later if this works at all.
    N is the number of neurons per committee member.
    """
    mems=[esn.simple_ESN(N=N,K=train_in[i].shape[1],L=train_in[i].shape[1],
                         a=a,binary_node=bin_node,feedback=fb,W_sig=W_sig,
                         in_sig=in_sig,fb_sig=fb_sig) for i in range(0,memnum)]
    for i in range(0,memnum):
        mems[i].generate_W(rho=rho,dens=dens)
        mems[i].generate_Win(dens=in_dens)
        if fb==1:
            mems[i].generate_Wfb(dens=fb_dens)
        mems[i].train_ESN(input_dat=train_in[i],teacher=train_out[i],around=around,order=order,washout=wo,noise=noise,
                  bias=bias,mp=MP,B=B)
        run_dims=run[i].shape
        mems[i].run_ESN(input_dat=run[i],around=around,order=order,time=run_dims[0],init='last')
    outs=[mems[i].outputs for i in range(0,memnum)]
    return outs

def Pixel_by_Pixel(train,run,N=500,fb=0,rho=1,W_sig=1,in_sig=1,fb_sig=1,
                dens=0.05,in_dens=0.8,fb_dens=0.1,a=1,noise=0.0,bias=1,bin_node=0,
                wo=100,MP=True,B=0.1,around=0,order=None):
    sh=train.shape
    n=sh[1]
    numin=int((n*n)/4)
    train_in=np.reshape(train[:,0:n:2,0:n:2],(sh[0],numin))
    train_out=np.reshape(train,(sh[0],int(n*n)))
    mems=[esn.simple_ESN(N=N,K=numin,L=1,a=a,feedback=fb,W_sig=W_sig,
                         in_sig=in_sig,fb_sig=fb_sig) for i in range(0,int(n*n))]
    output=np.zeros((run.shape[0],int(n*n)))
    for i in range(0,n*n):
        mems[i].generate_W(rho=rho,dens=dens)
        mems[i].generate_Win(dens=in_dens)
        if fb==1:
            mems[i].generate_Wfb(dens=fb_dens)
        train_outi=np.reshape(train_out[:,i],(sh[0],1))
        mems[i].train_ESN(input_dat=train_in,teacher=train_outi,around=around,
            order=order,washout=wo,noise=noise,bias=bias,mp=MP,B=B)
        mems[i].run_ESN(input_dat=run,around=around,order=order,time=run.shape[0],init='rand')
        print('Input reservoir '+str(i)+' trained')
        output[:,i]=np.reshape(mems[i].outputs,(run.shape[0]))


    output=np.reshape(output,(run.shape[0],n,n))
    return output


def Pixel_Avgd(train,run,N=500,fb=0,rho=1,W_sig=1,in_sig=1,fb_sig=1,
                dens=0.05,in_dens=0.8,fb_dens=0.1,a=1,noise=0.0,bias=1,bin_node=0,
                wo=100,MP=True,B=0.1,around=0,order=None):
    sh=train.shape
    n=sh[1]
    numin=int((n*n)/4)
    train_in=np.reshape(train[:,0:n:2,0:n:2],(sh[0],numin))
    train_out=np.reshape(train,(sh[0],int(n*n)))
    mems=[esn.simple_ESN(N=N,K=numin,L=1,a=a,feedback=fb,W_sig=W_sig,
                         in_sig=in_sig,fb_sig=fb_sig) for i in range(0,int(n*n))]
    mems[0].generate_W(rho=rho,dens=dens)
    mems[0].generate_Win(dens=in_dens)
    if fb==1:
        mems[0].generate_Wfb(dens=fb_dens)
    output=np.zeros((run.shape[0],int(n*n)))
    Wout_avg=np.zeros((1,N))
    for i in range(0,int(n*n)):
        mems[i].W=mems[0].W
        mems[i].W_in=mems[0].W_in
        if fb==1:
            mems[i].W_fb=mems[0].W_fb
        train_outi=np.reshape(train_out[:,i],(sh[0],1))
        mems[i].train_ESN(input_dat=train_in,teacher=train_outi,around=around,
            order=order,washout=wo,noise=noise,bias=bias,mp=MP,B=B)
        print('Input reservoir '+str(i)+' trained')
        Wout_avg+=mems[i].W_out
    Wout_avg=Wout_avg/(n*n)
    for i in range(0,int(n*n)):
        mems[i].W_out=Wout_avg
        mems[i].run_ESN(input_dat=run,around=around,order=order,time=run.shape[0],init='rand')
        output[:,i]=np.reshape(mems[i].outputs,(run.shape[0]))


    output=np.reshape(output,(run.shape[0],n,n))
    return output

def BlockRC(train,run,N=500,fb=0,rho=1,W_sig=1,in_sig=1,fb_sig=1,
                dens=0.05,in_dens=0.8,fb_dens=0.1,a=1,noise=0.0,bias=1,bin_node=0,
                wo=100,MP=True,B=0.1,around=0,order=None):
    """
    For lattices with fixed bounaries. Must have a lattice size that is a multiple
    of 4.
    """
    if (train is None) or (run is None):
        raise ValueError('no input or training data passed')
    tsh=train.shape
    num=int((tsh[1]/4)**2)
    numsqrt=int(mt.sqrt(num))
    #mask=np.ones((4,4),dtype=bool)
    #mask[0:4:2,0:4:2]=False
    trlist=[]
    outs=[]
    for i in range(0,numsqrt):
        for j in range(0,numsqrt):
            trlist.append(train[:,4*i:(4*i+4),4*j:(4*j+4)])
    tr_inlist=[np.reshape(trlist[i][:,0:4:2,0:4:2],(tsh[0],4)) for i in range(0,num)]
    tr_outlist=[np.reshape(trlist[i],(tsh[0],16)) for i in range(0,num)]
    mems=[esn.simple_ESN(N=N,K=4,L=16,a=a,feedback=fb,W_sig=W_sig,
                         in_sig=in_sig,fb_sig=fb_sig) for i in range(0,num)]
    for i in range(0,num):
        mems[i].generate_W(rho=rho,dens=dens)
        mems[i].generate_Win(dens=in_dens)
        if fb==1:
            mems[i].generate_Wfb(dens=fb_dens)
        mems[i].train_ESN(input_dat=tr_inlist[i],teacher=tr_outlist[i],around=around,
            order=order,washout=wo,noise=noise,bias=bias,mp=MP,B=B)
        mems[i].run_ESN(input_dat=run[i],around=around,order=order,time=run[i].shape[0],init='last')
        outs.append(np.reshape(mems[i].outputs,(run[i].shape[0],4,4)))
        print('Block number '+str(i)+' completed.')
    outarr=np.zeros((run[0].shape[0],tsh[1],tsh[2]))
    for i in range(0,numsqrt):
        for j in range(0,numsqrt):
            outarr[:,4*i:(4*i+4),4*j:(4*j+4)]=outs[int((numsqrt*i)+j)]
    return outarr

def SingleBlock(tr_in=None,tr_out=None,N=500,fb=0,rho=1,W_sig=1,in_sig=1,fb_sig=1,
                dens=0.05,in_dens=0.8,fb_dens=0.1,a=1,noise=0.0,bias=1,bin_node=0,
                wo=100,MP=True,B=0.1,around=0,order=None):
    """
    For training a single RC block using multiprocessing
    """
    rc=esn.simple_ESN(N=N,K=4,L=16,a=a,feedback=fb,W_sig=W_sig,in_sig=in_sig,fb_sig=fb_sig)
    rc.generate_W(rho=rho,dens=dens)
    rc.generate_Win(dens=in_dens)
    if fb==1:
        rc.generate_Wfb(dens=fb_dens)
    rc.train_ESN(input_dat=tr_in,teacher=tr_out,around=around,order=order,
                 washout=wo,noise=noise,bias=bias,mp=MP,B=B)
    W=rc.W
    W_in=rc.W_in
    W_out=rc.W_out
    rdict={'W':W,'W_in':W_in,'W_out':W_out}
    if fb==1:
        rdict['W_fb']=rc.W_fb
    return rdict

def feed_to_SingleBlock(indict):
    rdict=SingleBlock(**indict)
    return rdict

"""
def multi_BlockRC(train,parmdict):
    tsh=train.shape
    num=int((tsh[1]/4)**2)
    numsqrt=int(mt.sqrt(num))
    trlist=[]
    for i in range(0,numsqrt):
        for j in range(0,numsqrt):
            trlist.append(train[:,4*i:(4*i+4),4*j:(4*j+4)])
    tr_inlist=[np.reshape(trlist[i][:,0:4:2,0:4:2],(tsh[0],4)) for i in range(0,num)]
    tr_outlist=[np.reshape(trlist[i],(tsh[0],16)) for i in range(0,num)]
    dictlist=[copy.copy(parmdict) for i in range(0,num)]
    for i in range(0,num):
        dictlist[i]['tr_in']=tr_inlist[i]
        dictlist[i]['tr_out']=tr_outlist[i]
    try:
        workers=cpu_count()
    except NotImplementedError:
        print('Cant get CPU number. Using 12, assuming this is running on the devbox')
        workers=12
    pool=Pool(processes=workers)
    results=pool.map(feed_to_SingleBlock,dictlist,chunksize=6)
    print('Pool done')
    for i in range(0,len(results)):
        filename='/raid/shofer/expsaves/RCs/Wmats'+str(i+1)+'.mat'
        sio.savemat(filename,results[i])
    return None
"""
def BlockRC_2Layer(train,run,nn=1,N1=500,fb1=0,rho1=1,W_sig1=1,in_sig1=1,fb_sig1=1,
                dens1=0.05,in_dens1=0.8,fb_dens1=0.1,a1=1,noise1=0.0,bias1=1,
                wo1=100,MP1=True,B1=0.1,around1=0,order1=None,N2=500,
                fb2=0,rho2=1,W_sig2=1,in_sig2=1,fb_sig2=1,dens2=0.05,in_dens2=0.8,
                fb_dens2=0.1,a2=1,noise2=0.0,bias2=1,wo2=100,MP2=True,B2=0.1,
                around2=0,order2=None):
    tsh=train.shape
    num=int((tsh[1]/4)**2)
    numsqrt=int(mt.sqrt(num))
    trlist=[]
    outs=[]
    for i in range(0,numsqrt):
        for j in range(0,numsqrt):
            trlist.append(train[:,4*i:(4*i+4),4*j:(4*j+4)])
    tr_inlist=[np.reshape(trlist[i][:,0:4:2,0:4:2],(tsh[0],4)) for i in range(0,num)]
    tr_outlist=[np.reshape(trlist[i],(tsh[0],16)) for i in range(0,num)]
    mems=[esn.simple_ESN(N=N1,K=4,L=16,a=a1,feedback=fb1,W_sig=W_sig1,
                         in_sig=in_sig1,fb_sig=fb_sig1) for i in range(0,num)]
    for i in range(0,num):
        mems[i].generate_W(rho=rho1,dens=dens1)
        mems[i].generate_Win(dens=in_dens1)
        if fb1==1:
            mems[i].generate_Wfb(dens=fb_dens1)
        mems[i].train_ESN(input_dat=tr_inlist[i],teacher=tr_outlist[i],around=around1,
            order=order1,washout=wo1,noise=noise1,bias=bias1,mp=MP1,B=B1)
        mems[i].run_ESN(input_dat=tr_inlist[i],around=around1,order=order1,time=tsh[0],init='rand')
        outs.append(np.reshape(mems[i].outputs[wo1:,:],((tsh[0]-wo1),4,4)))
        print('Block number '+str(i)+' completed.')
    outarr=np.zeros(((tsh[0]-wo1),tsh[1],tsh[2]))
    for i in range(0,numsqrt):
        for j in range(0,numsqrt):
            outarr[:,4*i:(4*i+4),4*j:(4*j+4)]=outs[int((numsqrt*i)+j)]

    #Moore RC's
    padarr=np.pad(outarr,((0,0),(nn,nn),(nn,nn)),'constant',constant_values=0)
    mtrain=[]
    mtrain_out=[]
    nodenum=int(tsh[1]*tsh[2])
    in_dim=int((2*nn)+1)
    for i in range(nn,(nn+tsh[1])):
        for j in range(nn,(nn+tsh[1])):
            mtrain.append(np.reshape(padarr[:,(i-nn):(i+nn+1),(j-nn):(j+nn+1)],(padarr.shape[0],int(in_dim*in_dim))))
            mtrain_out.append(np.reshape(padarr[:,i,j],(padarr.shape[0],1)))
    moorercs=[esn.simple_ESN(N=N2,K=int(in_dim*in_dim),L=1,a=a2,feedback=fb2,W_sig=W_sig2,
                             in_sig=in_sig2,fb_sig=fb_sig2) for i in range(0,nodenum)]
    for i in range(0,nodenum):
        moorercs[i].generate_W(rho=rho2,dens=dens2)
        moorercs[i].generate_Win(dens=in_dens2)
        if fb2==1:
            moorercs[i].generate_Wfb(dens=fb_dens2)
        moorercs[i].train_ESN(input_dat=mtrain[i],teacher=mtrain_out[i],around=around2,
            order=order2,washout=wo2,noise=noise2,bias=bias2,mp=MP2,B=B2)
        print('Moore RC number '+str(i)+' completed.')

    #RCs trained, now run.
    outsr=[]
    for i in range(0,num):
        mems[i].run_ESN(input_dat=run[i],around=around1,order=order1,time=run[i].shape[0],init='last')
        outsr.append(np.reshape(mems[i].outputs,(run[i].shape[0],4,4)))
    print('Block RC run complete')
    outarr2=np.zeros((run[0].shape[0],tsh[1],tsh[2]))
    for i in range(0,numsqrt):
        for j in range(0,numsqrt):
            outarr2[:,4*i:(4*i+4),4*j:(4*j+4)]=outsr[int((numsqrt*i)+j)]
    padarr2=np.pad(outarr2,((0,0),(nn,nn),(nn,nn)),'constant',constant_values=0)
    mrun=[]
    finalout=[]
    for i in range(nn,(nn+tsh[1])):
        for j in range(nn,(nn+tsh[1])):
            mrun.append(np.reshape(padarr2[:,(i-nn):(i+nn+1),(j-nn):(j+nn+1)],(run[0].shape[0],int(in_dim*in_dim))))
    for i in range(0,nodenum):
        moorercs[i].run_ESN(input_dat=mrun[i],around=around2,order=order2,time=run[0].shape[0],init='last')
        finalout.append(np.reshape(moorercs[i].outputs,(run[0].shape[0])))
    print('Moore RC run complete')
    outarrf=np.zeros((run[0].shape[0],tsh[1],tsh[2]))
    for i in range(0,tsh[1]):
        for j in range(0,tsh[2]):
            outarrf[:,i,j]=finalout[int((tsh[1]*i)+j)]
    return outarrf

def get_numbers(filename):
    num=re.findall(r'[+-]?\d+(?:\.\d+)?',filename)
    numf=[float(i) for i in num]
    #if len(numf)==1:
    #    numf=int(num[0])
    if numf==[]:
        try:
            splt=filename.split('.')
            names=splt[0].split('_')
            numf=names+numf
        except:
            pass
    return numf


def get_data(directory,ax=1):
    """
    returns an array with:
        [parameter val,avg0,std0,avg1,std1,...]
    """
    full_list=[]
    for root,dirs,files in os.walk(directory):
        for file in files:
            if not (file.startswith('.') or file.startswith('x')):
                num=get_numbers(file)
                data=np.genfromtxt(directory+'/'+file,delimiter=',',dtype=np.float64)
                avg=np.average(data,axis=ax)
                dims=avg.shape
                std=np.std(data,axis=ax)
                datlist=num
                for j in range(0,dims[0]):
                    datlist.extend([avg[j],std[j]])
                full_list.append(datlist)
    full_arr=np.asarray(full_list)
    return full_arr

def get_data_cml(directory):
    """
    returns an array with:
        [parameter val,avg0,std0,avg1,std1,...]
    """
    full_list=[]
    for root,dirs,files in os.walk(directory):
        for file in files:
            if not (file.startswith('.') or file.startswith('x')):
                num=get_numbers(file)
                data=np.genfromtxt(directory+'/'+file,delimiter=',',dtype=np.float64)
                avg=np.average(data,axis=None)
                std=np.std(data,axis=None)
                datlist=num
                datlist.extend([avg,std])
                full_list.append(datlist)
    full_arr=np.asarray(full_list)
    return full_arr

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

def LempelZivComplexity(arr):
    arrs=arr.tostring()
    arrsl=float(len(arrs))
    com=zl.compress(arrs,9)
    coms=float(len(com))
    ratio=coms/arrsl
    return ratio

def AvgLempelZiv_byFrame(arr):
    lis=[]
    al=arr.shape[0]
    for i in range(0,al):
        arri=arr[i,:,:]
        comi=LempelZivComplexity(arri)
        lis.append(comi)
    avg=sum(lis)/len(lis)
    return avg
