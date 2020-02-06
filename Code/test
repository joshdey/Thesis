from ESN import simple_ESN as sesn
import ESN
import ESNteachers as teach
import numpy as np

lor = sesn(N=400, K=1, L=2, binary_node=0)
W = lor.generate_W(rho=0.9, dens=0.49, cont=0)
W = lor.specify_W(W, rho=0.9)
W=lor.rescale_W(W)
Win=lor.generate_Win()
W_fb = lor.generate_Wfb()

x, y, z = teach.Lorenz(10000, 0.01)
yz = np.concatenate((y,z), axis=1)

train = lor.train_ESN(input_dat=x, teacher=yz, around=.01, order=2, washout=100, mp=False, B=0.01)
testrun, yz = lor.run_ESN(input_dat=x,around=.01, order=2,time=10000)

lorattractor=lor.plot_attractor(name='testLorenz')
