"""
@author: joshdey
"""
import numpy as np
import random as rn
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import networkx as nx
import mpmath as mpm
import auxiliary as aux


class simple_ESN:
    """
    The basic reservoir computer (echo state network) object class.

    Attributes:
        N (int): number of neurons
        K (int): number of external inputs
        L (int): number of outputs
        a (float): leak rate
        W (NxN array): internal neuron weight array
        W_in (NxK array): input weight array
        W_out (Lx(K+N) array): output weight array
        W_fb (NxL array): feedback weight array
        binary_node (boolean): 0 for internal weights to be continuously
            distributed
        feedback (boolean): 0 for no feedback, 1 for feedback
        W_sig (float): internal weights are drawn from (-W_sig, W_sig). Note
            that the weights are rescaled to (-rho,rho) later.
        in_sig (float): input weights are drawn from (-in_sig, in_sig).
        fb_sig (float): input weights are drawn from (-fb_sig, fb_sig).
        directed (boolean): 0 for symmetric internal weight matrix, (i.e.
            neurons are non-directional), 1 for asymmetric internal weight
            matrix.
    """

    def __init__(self,N=100,K=1,L=2,a=1,binary_node=0,feedback=1,
                 W_sig=1,in_sig=1,fb_sig=1,directed=True):
        self.N=N #number of internal neurons (dim of state vec)
        self.K=K #number of external inputs (dim of input vec)
        self.L=L #number of outputs (dim of output vec)
        self.a=a #leak rate (no leak=1, total leak=0)

        self.W=np.zeros([N,N]) #initialize all weight arrays
        self.W_in=np.zeros([N,K]) #initialize all weight arrays
        self.W_out=np.zeros([L,(K+N)]) #initialize all weight arrays
        self.W_fb=np.zeros([N,L]) #initialize all weight arrays

        self.binary_node=binary_node
        self.feedback=feedback
        self.W_sig=W_sig
        self.in_sig=in_sig
        self.fb_sig=fb_sig
        self.directed=directed


    def distribution(self,binary=0,sig=1):
        """
        Function to provide a uniformly distributed pseudorandom number.

        Args:
            self (simple_ESN object): the reservoir computer object.
            binary (boolean): 0 for continuous distribution, 1 for a "binary"
                number (i.e. 50% chance of -sig, 50% chance of sig).
            sig (float): random number is drawn from (-sig, sig).
        Returns:
            u: pseudorandom number.
        """
        if binary==1:
            p=np.random.random()
            if p<0.5:
                u=-sig
            else:
                u=sig
        else:
            u=rn.uniform(-sig,sig)
        return u



    def generate_W(self,rho=0.8,dens=0.49,edge=None,cont=True):
        """
        Generates the internal weight matrix W for a simple_ESN object.

        Args:
            self (simple_ESN object): the reservoir computer object.
            rho (float): the desired spectral radius of the weight matrix W.
            dens (float): the desired density (fraction of nonzero entries) of
                W.
            edge (int): if edge is not None, generates a weight matrix with a
                specified number of edges (given by edge).
            cont (boolean): 0 for "binary" weights (i.e. weights are either
                -W_sig or W_sig before rescaling to rho).
        Yields:
            self.W (NxN array): the internal weight matrix for the simple_ESN
                object.
        """
        self.rho=rho
        self.W_dens=dens

        if edge is not None:
            g=nx.gnm_random_graph(self.N,edge,directed=self.directed)
        else:
            g=nx.gnp_random_graph(self.N,dens,directed=self.directed)
        Asp=nx.to_scipy_sparse_matrix(g,dtype=np.float64)
        W=Asp.toarray()

        if cont is True:
            for i in range(0,self.N):
                for j in range(0,self.N):
                    if W[i,j]==1:
                        W[i,j]=self.distribution(binary=self.binary_node,
                        sig=self.W_sig)
        else:
            for i in range(0,self.N):
                for j in range(0,self.N):
                    if W[i,j]==1:
                        W[i,j]=self.distribution(binary=self.binary_node,
                        sig=self.W_sig)

        self.raw_W=W
        eigval=np.linalg.eigvals(W)
        self.eigval=eigval
        lam=np.amax(np.abs(eigval))
        W=(rho/lam)*W
        self.norm_eigval=(rho/lam)*eigval
        self.W=W
        return W


    def specify_W(self,W=None,rho=0.8):
        """
        Allows a specific adjacency matrix to be associated to a simple_ESN
        object,and rescaled to a given spectral radius rho.

        Args:
            self (simple_ESN object): the reservoir computer object.
            W (NxN array): unscaled neuron connection matrix.
            rho (float): specified spectral radius.
        Yields:
            self.W (NxN array): the internal weight matrix for the simple_ESN
                object.
        Returns:
            W
        """
        try:
            self.N=W.shape[0]
            eigval,eigvec=np.linalg.eig(W)
            lam=abs(np.amax(eigval))
            W=(rho/lam)*W
            self.W=W
        except:
            raise ValueError("Weight matrix must be numpy array")
        return W

    def rescale_W(self,new_rho=1):
        """
        Rescales an existing internal weight matrix.

        Args:
            self (simple_ESN object): the reservoir computer object.
            new_rho (float): new spectral radius for the internal weight matrix.
        Yields:
            self.W (NxN array): the rescaled internal weight matrix.
        """
        W=self.W
        W=(new_rho/self.rho)*W
        self.rho=new_rho
        self.W=W
        return W


    def generate_Win(self,dens=0.49):
        """
        Generates the input weight matrix for a simple_ESN object.

        Args:
            self (simple_ESN object): the reservoir computer object.
            dens (float): the desired density (fraction of nonzero entries) of
                W_in.
        Yields:
            self.W_in (NxK array): the input weight matrix.
        """
        W_in=np.zeros([self.N,self.K])
        self.Win_dens=dens
        for i in range(0,self.N):
            for j in range(0,self.K):
                p=np.random.random()
                if p<dens:
                    W_in[i,j]=self.distribution(binary=False,sig=self.in_sig)
        self.W_in=W_in
        return W_in


    def generate_Wfb(self,dens=0.49):
        """
        Generates the feedback weight matrix for a simple_ESN object.

        Args:
            self (simple_ESN object): the reservoir computer object.
            dens (float): the desired density (fraction of nonzero entries) of
                W_fb.
        Yields:
            self.W_fb (NxL array): the feedback weight matrix.
        """
        W_fb=np.zeros([self.N,self.L])
        self.Wfb_dens=dens
        if self.feedback==1:
            for i in range(0,self.N):
                for j in range(0,self.L):
                    p=np.random.random()
                    if p<dens:
                        W_fb[i,j]=self.distribution(binary=False,
                            sig=self.fb_sig)
        self.W_fb=W_fb
        return W_fb


    def train_ESN(self,input_dat=None,teacher=None,around=0,order=None,
        washout=100, noise=None,bias=0,mp=True,B=0):
        """
        Single shot training of a simple_ESN object. Generates W_out.

        Args:
            self (simple_ESN object): the reservoir computer object.
            input_dat (TxK array): the training input data. T is the temporal
                length of the data, K is the number of inputs.
            teacher (TxL array): the training output ("teacher") data. T is the
                temporal length of the data, L is the number of outputs.
            around (array or float): provides coefficients to neuron activation
                functions.
            order (str,int): 'power' gives a Taylor expansion with coefficients
                given by around, 'relu' gives a rectified linear activation
                function with input scaling around, an integer gives the Taylor
                expansion of tanh about around, while any non integer,
                non-string value gives the standard tanh activation function.
            washout (int): number of discarded initial transient steps.
            noise (float): arbitrary noise added to each training update step.
            bias (float): bias added to each update step.
            mp (Boolean True/False): True for Moore-Penrose pseudoinverse,
                False for ridge regression.
            B (float): ridge regression parameter.
        Yields:
            self.W_out (LxN array): trained simple_ESN output weights.
        """
        time=teacher.shape[0] #(time,L)
        if input_dat is None:
            input_dat=np.zeros([time,self.K])
        if order=='power':
            coeffs=np.array(around)
            update=aux.taylor_exp
        elif order=='relu':
            if around==0:
                coeffs=1
            else:
                coeffs=around
            update=aux.ReLU
        elif isinstance(order,int) is True:
            mpm.dps=16
            mpm.pretty=True
            coeffs=np.array(mpm.chop(mpm.taylor(mpm.tanh,around,order)),
                dtype=np.float64)
            update=aux.taylor_exp
        else:
            coeffs=None
            update=np.tanh
        M=np.zeros([time-washout,self.N]) #state collecting matrix
        T=np.zeros([time-washout,self.L]) #target outputs
        x=np.zeros([self.N])
        y=np.zeros([self.L])

        a=self.a
        self.bias=bias
        for t in range(0,time):
            u=input_dat[t,:]
            if noise is not None:
                v=rn.uniform(-noise,noise)
            else:
                v=0
            x=(1-a)*x + a*update(np.dot(self.W,x)+
               np.dot(self.W_in,u)+np.dot(self.W_fb,y)+v+bias,coeffs)
            if t>=washout:
                k=t-washout
                M[k,:]=x #just use internal states for now
                T[k,:]=teacher[t,:]
            y=teacher[t,:]

        #Set output weights
        self.M=M
        self.T=T
        if mp is True: #Moore-Penrose pseudoinverse
            W_out=np.dot(np.linalg.pinv(M),T)
            W_out=np.transpose(W_out)
        else: #Ridge Regression
            sq=np.dot(np.transpose(M),M)
            inv=np.linalg.inv(sq + B*np.identity(sq.shape[0]))
            W_out=np.dot(np.dot(np.transpose(T),M),inv)
        self.W_out=W_out
        return W_out



    def run_ESN(self,input_dat=None,around=0,order=None,time=200,init='last',
        force_start=None):
        """
        Runs the resevoir computer for a given number of time steps, with or
        without input data.

        Args:
            self (simple_ESN object): the reservoir computer object.
            input_dat (timexK array or None): the input data. T is the temporal
            length of the data, K is the number of inputs.
            around (array or float): provides coefficients to neuron activation
                functions.
            order (str,int): 'power' gives a Taylor expansion with coefficients
                given by around, 'relu' gives a rectified linear activation
                function with input scaling around, an integer gives the Taylor
                expansion of tanh about around, while any non integer,non-string
                value gives the standard tanh activation function.
            time (int): number of steps to run the reservoir.
            init (str): 'last' starts the run at the last iteration of the
                training run. 'zero' starts the run with all neurons set to
                zero, and any other value starts the run with neuron
                displacements randomize.
            force_start (anything): if force_start is not None, starts output
                at last value of the teacher data.
        Yields:
            self.states (timexN array): states of the reservoir during the run.
            self.outputs (timexL array): reservoir outputs for run.
        """
        if input_dat is None:
            input_dat=np.zeros([time,self.K])
        if init=='last':
            x=self.M[-1,:]
        elif init=='zero':
            x=np.zeros([self.N])
        else:
            x=np.random.random_sample([self.N])
        if order=='power':
            coeffs=np.array(around)
            update=aux.taylor_exp
        elif order=='relu':
            if around==0:
                coeffs=1
            else:
                coeffs=around
            update=aux.ReLU
        elif isinstance(order,int) is True:
            mpm.dps=16
            mpm.pretty=True
            coeffs=np.array(mpm.chop(mpm.taylor(mpm.tanh,around,order)),
                dtype=np.float64)
            update=aux.taylor_exp
        else:
            coeffs=None
            update=np.tanh
        if force_start is not None:
            y=force_start #should be last time step of teacher
        else:
            y=np.zeros([self.L])
        a=self.a

        bias=self.bias

        states=np.zeros([time,self.N])
        outputs=np.zeros([time,self.L])

        for t in range(0,time):
            u=input_dat[t,:]
            x=(1-a)*x + a*update(np.dot(self.W,x)+
               np.dot(self.W_in,u)+np.dot(self.W_fb,y)+bias,coeffs)
            y=np.dot(self.W_out,x)

            states[t,:]=x
            outputs[t,:]=y
        self.input=input_dat
        self.states=states
        self.outputs=outputs
        return states, outputs

    def plot_internal(self,nodes=[0,5],rang=True,times=[100,300],train=True,
        pltnum=1,pltsize=(12,5)):
        """
        Plots the internal states of some (or all) of the reservoir neurons
        against time.

        Args:
            self (simple_ESN object): the reservoir computer object.
            nodes (list or tuple): the neurons to be plotted. If nodes is a
                tuple and rang is True, plots all nodes in the range of the
                tuple.
            rang (True/False): if rang is true, treats nodes as a tuple defining
                a range of neurons.
            times (tuple of ints): times to plot the nodes.
            train (True/False): set to True to display training states.
            pltnum (int): figure number for output.
            pltsize (tuple of floats): size of figure.
        Yields:
            Plots of neuron states.
        """
        if rang is True:
            nodes=list(range(nodes[0],nodes[1]+1))
        if train is True:
            data=self.M
        else:
            data=self.states
        plt.figure(num=pltnum,figsize=pltsize)
        for i in nodes:
            plt.plot(data[times[0]:times[1],i])


    def plot_nodes(self,cols=4,rows=10,times=[0,250],pltnum=1,pltsize=(20,30)):
        """
        Plots all internal neurons of a reservoir computer.

        Args:
            self (simple_ESN object): the reservoir computer object.
            cols (int): number of columns in the plot.
            rows (int): number of rows in the plot.
            times (tuple of ints): times to plot the neuron states.
            pltnum (int): plot number.
            pltsize (tuple of floats): size of the plot.
        Yields:
            Single plot of neurons states.
        """
        data=self.states
        fig,axs=plt.subplots(nrows=rows,ncols=cols,num=pltnum,figsize=pltsize)
        s=0
        for i in range(0,rows):
            for j in range(0,cols):
                axs[i,j].plot(data[times[0]:times[1],s+j])
            s+=cols


    def plot_attractor(self,in_ax=0,axes=[0,1],fb=False,save=False,filepath=None,
        name=None):
        """
        Plots the input and outputs of the reservoir computer against each
        other. Requires a 1-input, 2-output reservoir.

        Args:
            self (simple_ESN object): the reservoir computer object.
            in_ax (int): input data axis.
            axes (tuple of ints): output data axes.
            fb (True/False): True for networks with no input, just using
            feedback outputs.
            save (True/False): saves plot if True, doesn't otherwise.
            loc (str): save location for plot.
            name (str): filename for plot.
        Yields:
            Attractor plot of resevoir input/output. Saves plot if save is True.
        """
        if filepath is None:
            filepath='/Users/joshdey/Documents/GitHub/Thesis/Code/AttractorSaves/'
        file = filepath+name
        fig=plt.figure(num=2,figsize=(9,9))
        ax=fig.add_subplot(111,projection='3d')
        if fb is False:
            ax.plot(self.input[:,in_ax],self.outputs[:,axes[0]],
                self.outputs[:,axes[1]])
        else:
            ax.plot(self.outputs[:,axes[0]],self.outputs[:,axes[1]],
                self.outputs[:,axes[2]])
        plt.rc('text', usetex=True)
        ax.set_xlabel(r'$\tilde x$',fontsize=24)
        ax.set_ylabel(r'$\tilde y$',fontsize=24)
        ax.set_zlabel(r'$\tilde z$',fontsize=24)
        ax.tick_params(axis='both',which='major',labelsize=0)
        plt.rc('text', usetex=False)
        if save is True:
            plt.savefig(file,dpi=300)



    def plot_eigvals(self,norm=True,pltnum=1,pltsize=(8,8)):
        """
        Plots eigenvalues of the internal weight matrix.

        Args:
            self (simple_ESN object): the reservoir computer object.
            norm (True/False): plots the non-rescaled to rho eigenvalues if
            norm is False.
            pltnum (int): plot number.
            pltsize (tuple of floats): size of the plot.
        Yields:
            Plot of the eigenvalues of the internal weight matrix.
        """
        plt.figure(num=pltnum,figsize=pltsize)
        if norm is False:
            plt.plot(np.real(self.eigval),np.imag(self.eigval),'.')
        else:
            plt.plot(np.real(self.norm_eigval),np.imag(self.norm_eigval),'.')


    def save_ESN(self,filename,filepath=None):
        """
        Saves the attributes of a simple_ESN object.

        Args:
            self (simple_ESN object): the reservoir computer object.
            filename (str): name of the file.
            filepath (str): location of the file.
        Yields:
            simple_ESN object saved as a .npz file.
        """
        if filepath is None:
            filepath='/Users/joshdey/Documents/GitHub/Thesis/Code/ESNsaves/'
        file=filepath+filename
        meta=[self.N,self.K,self.L,self.a,self.binary_node,self.feedback,
              self.W_sig,self.in_sig,self.fb_sig,self.rho,self.W_dens,
              self.Win_dens,self.Wfb_dens]
        meta=np.array(meta)
        np.savez(file,meta=meta,W=self.W,Win=self.W_in,Wfb=self.W_fb,
            Wout=self.W_out)


def load_ESN(file):
    """
    Loads a simple_ESN object from a .npz file.

    Args:
        file (str): filename and path of the simple_ESN save.
    Returns:
        simple_ESN object with filled attributes.
    """
    mats=np.load(file)
    meta=mats['meta']
    ESN=simple_ESN(N=int(meta[0]),K=int(meta[1]),L=int(meta[2]),a=meta[3],
        binary_node=meta[4], feedback=meta[5],W_sig=meta[6],in_sig=meta[7],
        fb_sig=meta[8])
    ESN.W=mats['W']
    ESN.W_in=mats['Win']
    ESN.W_fb=mats['Wfb']
    ESN.W_out=mats['Wout']

    return ESN
