# We modified "Karcher Mean.ipynb" in https://github.com/aorvieto/RAGDsDR (Alimisis et al., 2021)

import os
import numpy as np
import numpy.linalg as la
from scipy.linalg import sqrtm, logm, expm
from utils import random_psd, karcher_mean
import matplotlib.pyplot as plt
from optimizers import RGD_optimizer, RAGD_optimizer,RAGDsDR_optimizer, RNAG_C_optimizer, RNAG_SC_optimizer
from pymanopt.manifolds import PositiveDefinite

os.environ['KMP_DUPLICATE_LIB_OK']='True'
plt.rcParams['text.usetex'] = True
plt.rcParams["font.family"] = "serif"

# problem hyperparameters
seed = 1 #for reproducibility
d = 100 #problem dimension
kappa = 1e6 #condition number of matrices
n = 50 #number of matrices to average

# algorithm hyperparameters
accuracy = 1e-5 #accuracy of the numerical solution (Zhang and Sra 2016)

N=100 #number of iterations for optimizers

A = np.zeros((d,d,n)) #list of matrices to average
np.random.seed(seed) #
matrix_seeds = np.random.randint(0, 1000, n)
for i in range(n): A[:,:,i] = random_psd(d,kappa,matrix_seeds[i]) #initializing matrices at random
print('Computing numerical solution...')
x_sol = karcher_mean(A, accuracy) #basically doing RGD, but in an explicit form with no exp map.

M = PositiveDefinite(d)

def dist(X,Y): return M.dist(X,Y)

def log(X,Y): return M.log(X,Y)

def cost(X):
    c=0
    for i in range(n): c += (1/(2*n))*M.dist(A[:,:,i],X)**2
    return c

def grad(X): #see https://www.math.fsu.edu/~whuang2/pdf/VisitBJU_Slides.pdf
    c=0*X
    for i in range(n): c -= (1/n)*log(X,A[:,:,i])
    return c

def exp(X,U): # M.exp(X,U) by pymanopt is unstable!
    c = la.cholesky(X)
    c_inv = la.inv(c)
    e = expm(np.dot(np.dot(c_inv, U),c_inv.T))
    return np.dot(np.dot(c, e), c.T)

def transp(X,Y,U): return M.transp(X,Y,U)

#initialization
x0 = np.mean(A,axis=2)

L = 10
mu = 1 #the problem is a sum of scaled distances
zeta = 1

#running the optimizers
t1, x1, f1 = RGD_optimizer(N,x0,L,cost,grad,exp)
t2, x2, f2 = RAGDsDR_optimizer(N, x0, L, zeta, cost, grad, exp, log, transp, -1)
t3, x3, f3 = RAGD_optimizer(N, x0, L, mu, cost, grad, exp, log)
t4, x4, f4 = RNAG_C_optimizer(N, x0, L, zeta, cost, grad, exp, log, transp)
t5, x5, f5 = RNAG_SC_optimizer(N, x0, L, mu, zeta, cost, grad, exp, log, transp)

f_sol = cost(x_sol)

fig, ax = plt.subplots()
ax.loglog(range(N), f1-f_sol,label='RGD', color = 'gray')
ax.loglog(range(N), f2-f_sol, label=r'RAGDsDR', color='blue')
ax.loglog(range(N), f3-f_sol, label='RAGD (Zhang and Sra)', color='green')
ax.loglog(range(N), f4-f_sol, label='RNAG-C', color ='goldenrod')
ax.loglog(range(N), f5-f_sol,label='RNAG-SC', color = 'red')

plt.xlabel('Iterations',size=20)
plt.ylabel('$f(x_k)-f(x^*)$',size=20)
ax.grid()
ax.legend(fontsize=15)
plt.show()
filename = 'results/spd_final.eps'
fig.savefig(filename, format='eps')