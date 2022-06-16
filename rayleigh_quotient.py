# We modified "Rayleigh quotient.ipynb" in https://github.com/aorvieto/RAGDsDR (Alimisis et al., 2021)

import os
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
from optimizers import RGD_optimizer, RAGDsDR_optimizer, RAGD_optimizer, RNAG_C_optimizer, RNAG_SC_optimizer
from pymanopt.manifolds import Sphere

os.environ['KMP_DUPLICATE_LIB_OK']='True'
plt.rcParams['text.usetex'] = True
plt.rcParams["font.family"] = "serif"

# problem hyperparameters
seed = 1 #for reproducibility
d = 1000 #problem dimension

# algorithm hyperparameters
N = 30000 #number of iterations for optimizers

# Problem definition & analytic solution
np.random.seed(seed)
A = np.random.randn(d,d)/np.sqrt(d)
H = (A+A.T)/2
eigenvalues, eigenvectors = la.eig(H)
max_eig = np.max(eigenvalues)
min_eig = np.min(eigenvalues)
print('Computing numerical solution...')
x_sol = eigenvectors[:, np.argmax(eigenvalues)]/la.norm(eigenvectors[:, np.argmax(eigenvalues)])

#Manifold functions
M = Sphere(d)

def cost(X):return -0.5*np.dot(X,np.dot(H, X))
def egrad(X):return -np.dot(H,X)
def grad(X):return M.egrad2rgrad(X,egrad(X))
def retr(X):return M.retr(X,0*X)
def exp(X,U):return M.exp(X,U)
def rexp(X,U):return retr(M.exp(X,U)) #more stable than just exp!!
def log(X,Y):return M.log(X,Y)
def dist(X,Y):return M.dist(X,Y)
def transp(X,Y,U):return M.transp(X,Y,U)

# iterations to solve
L = max_eig - min_eig
h = 1/L #stepsize
x0 = retr(10*np.random.randn(d,)) #starting position
zeta = 1

#running the optimizers
t1, x1, f1 = RGD_optimizer(N,x0,L,cost,grad,exp)
t2, x2, f2 = RAGDsDR_optimizer(N,x0,L,zeta,cost,grad,exp,log,transp,-1)
t3, x3, f3 = RNAG_C_optimizer(N,x0,L,zeta,cost,grad,exp,log,transp)

f_sol = cost(x_sol)

fig, ax = plt.subplots()
ax.loglog(range(N), f1-f_sol, label='RGD', color='gray')
ax.loglog(range(N), f2-f_sol, label=r'RAGDsDR', color='blue')
ax.loglog(range(N), f3-f_sol, label='RNAG-C', color='goldenrod')
plt.xlabel('Iterations',size=20)
plt.ylabel('$f(x_k)-f(x^*)$',size=20)
ax.grid()
ax.legend(fontsize=15)
plt.show()
filename = 'results/rayleigh_final.eps'
fig.savefig(filename, format='eps')