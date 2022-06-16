# We modified "Karcher Mean.ipynb" in https://github.com/aorvieto/RAGDsDR (Alimisis et al., 2021)

import os
import numpy as np
import matplotlib.pyplot as plt
from optimizers import RGD_optimizer, RAGD_optimizer,RAGDsDR_optimizer, RNAG_C_optimizer, RNAG_SC_optimizer
from geomstats.geometry import hyperboloid

os.environ['KMP_DUPLICATE_LIB_OK']='True'
plt.rcParams['text.usetex'] = True
plt.rcParams["font.family"] = "serif"

# problem hyperparameters
seed = 1 #for reproducibility
d = 1000 #problem dimension
n = 10 #number of points to average

# algorithm hyperparameters
N=100 #number of iterations for optimizers

A = np.zeros((d+1,n)) #list of points to average
np.random.seed(seed) #

# initializing points in hyperbolic space at random
for i in range(n):
    A[0,i] = 1
    for j in range(d):
        A[j+1,i] = (1/(d**0.5))*np.random.randn()
        A[0,i] += (A[j+1,i])**2
    A[0,i] = A[0,i]**(1/2)

M1 = hyperboloid.HyperboloidMetric(dim=d)
M2 = hyperboloid.Hyperboloid(dim=d)

def dist(X,Y): return M1.dist(X, Y)

def log(X,Y): return M1.log(Y, X)

def cost(X):
    c=0
    for i in range(n): c += (1/(2*n)) * M1.dist(A[:, i], X) ** 2
    return c

def grad(X): #see https://www.math.fsu.edu/~whuang2/pdf/VisitBJU_Slides.pdf
    c=0*X
    for i in range(n): c -= (1/n)*log(X,A[:,i])
    return c

def exp(X,U): return M1.exp(U, X)

def transp(X,Y,U): # We can approximate parallel transport by projection, as in https://github.com/NicolasBoumal/manopt/blob/master/manopt/manifolds/hyperbolic/hyperbolicfactory.m
    return M2.to_tangent(U,Y)

#initialization
x0 = np.zeros((d+1))
x0[0]=1
L = 10
mu = 1 #the problem is a sum of scaled distances
zeta = 1

#running the optimizers
t1, x1, f1 = RGD_optimizer(N,x0,L,cost,grad,exp)
t2, x2, f2 = RAGDsDR_optimizer(N, x0, L, zeta, cost, grad, exp, log, transp, -1)
t3, x3, f3 = RAGD_optimizer(N, x0, L, mu, cost, grad, exp, log)
t4, x4, f4 = RNAG_C_optimizer(N, x0, L, zeta, cost, grad, exp, log, transp)
t5, x5, f5 = RNAG_SC_optimizer(N, x0, L, mu, zeta, cost, grad, exp, log, transp)

#If N is large enough, then x1[N-1] approximates the optimal solution.
x_sol = x1[N-1]
f_sol = cost(x_sol) + 1e-14

fig, ax = plt.subplots()
ax.loglog(range(N), f1-f_sol,label='RGD', color = 'gray')
ax.loglog(range(N), f2-f_sol, label=r'RAGDsDR', color='blue')
ax.loglog(range(N), f3-f_sol, label='RAGD (Zhang and Sra)', color='green')
ax.loglog(range(N), f4-f_sol, label='RNAG-C', color ='goldenrod')
ax.loglog(range(N), f5-f_sol,label='RAGD-SC', color = 'red', linestyle=(0, (5, 5)))

plt.xlabel('Iterations',size=20)
plt.ylabel('$f(x_k)-f(x^*)$',size=20)
ax.grid()
ax.legend(fontsize=15)
plt.show()
filename = 'results/hyperbolic_final.eps'
fig.savefig(filename, format='eps')