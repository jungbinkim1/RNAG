# We modified "Rayleigh quotient.ipynb" in https://github.com/aorvieto/RAGDsDR (Alimisis et al., 2021)

import os
import numpy as np
import numpy.linalg as la
import math
import matplotlib.pyplot as plt
from optimizers import RGD_optimizer, RAGDsDR_optimizer, RAGD_optimizer, RNAG_C_optimizer, RNAG_SC_optimizer, SIRNAG_C_optimizer, SIRNAG_SC_optimizer
from pymanopt.manifolds import Sphere

os.environ['KMP_DUPLICATE_LIB_OK']='True'
plt.rcParams['text.usetex'] = True
plt.rcParams["font.family"] = "serif"

# problem hyperparameters
seed = 1 #for reproducibility
d = 10 #problem dimension
T = 50 #for ODE, we consider the domain (0,T)

## Problem definition & analytic solution
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
mu = 0.1
x0 = retr(10*np.random.randn(d,)) #starting position
xi = 2

#iss = 1 / (integration step size) = 1 / sqrt(s)
iss_ode = 1000
iss1 = 10
iss2 = 100
iss3 = 1000

#note that t ~ k * sqrt(s)
#we approximate solution of ODE by SIRNAG (Alimisis et al., 2020)
t_ode, x_ode, f_ode = SIRNAG_SC_optimizer(iss_ode * T + 1, x0, 1 / iss_ode, mu, xi, cost, grad, exp, log, transp)
t1, x1, f1 = RNAG_SC_optimizer(math.floor(iss1 * T) + 1, x0, (iss1 ** 2), mu, xi, cost, grad, exp, log, transp)
t2, x2, f2 = RNAG_SC_optimizer(iss2 * T + 1, x0, (iss2 ** 2), mu, xi, cost, grad, exp, log, transp)
t3, x3, f3 = RNAG_SC_optimizer(iss3 * T + 1, x0, (iss3 ** 2), mu, xi, cost, grad, exp, log, transp)

#generate the list [k * sqrt{s}]
list1 = []
for i in range(math.floor(iss1 * T) + 1):
    list1.append(i/iss1)

list2 = []
for i in range(iss2 * T + 1):
    list2.append(i/iss2)

list3 = []
for i in range(iss3 * T + 1):
    list3.append(i/iss3)

list_ode = []
for i in range(iss_ode * T + 1):
    list_ode.append(i/iss_ode)

fig, ax = plt.subplots()
ax.plot(list_ode, f_ode - cost(x_sol), label='ODE', color='gray')
ax.plot(list1, f1 - cost(x_sol), label=r'$s={10}^{-2}$', color ='gold')
ax.plot(list2, f2 - cost(x_sol), label=r'$s={10}^{-4}$', color ='orange')
ax.plot(list3, f3 - cost(x_sol), label=r'$s={10}^{-6}$', color ='red', linestyle=(0, (5, 5)))
ax.set_yscale('log')
plt.xlabel(r'$t=\sqrt{s}k$',size=20)
plt.ylabel('$f(x_k)-f(x^*)$',size=20)
ax.grid()
ax.legend(fontsize=15)
plt.show()

filename='results/strongly_convex_graph_final.eps'
fig.savefig(filename, format='eps')

plt.close()

#to compare algorithms with different integration step size
ratio41 = math.floor(iss_ode / iss1)
ratio42 = math.floor(iss_ode / iss2)
ratio43 = math.floor(iss_ode / iss3)

distance1 = []
for i in range(math.floor(iss1 * T) + 1):
    distance1.append(dist(x_ode[ratio41 * i], x1[i]))

distance2 = []
for i in range(iss2 * T + 1):
    distance2.append(dist(x_ode[ratio42 * i], x2[i]))

distance3 = []
for i in range(iss3 * T + 1):
    distance3.append(dist(x_ode[ratio43 * i], x3[i]))

fig, ax = plt.subplots()
ax.plot(list1, distance1,label=r'$s={10}^{-2}$', color='gold')
ax.plot(list2, distance2,label=r'$s={10}^{-4}$', color='orange')
ax.plot(list3, distance3,label=r'$s={10}^{-6}$', color='red')
plt.xlabel(r'$t=\sqrt{s}k$',size=20)
plt.ylabel(r'$d(x_k,X(t))$',size=20)
ax.grid()
ax.set_yscale('log')
ax.legend(fontsize=15)
plt.show()
filename='results/strongly_convex_distance_final.eps'
fig.savefig(filename, format='eps')