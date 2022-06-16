# We modified "geometric_optimizers.py" in https://github.com/aorvieto/RAGDsDR (Alimisis et al., 2021)

import time
import numpy as np
import numpy.linalg as la
from scipy.linalg import sqrtm, logm, expm

#This function is taken from "geometric_optimizers.py" in https://github.com/aorvieto/RAGDsDR (Alimisis et al., 2021)
#Classic RGD optimizer, for rates, check http://proceedings.mlr.press/v49/zhang16b.pdf
def RGD_optimizer(K,x0,L,cost,grad,exp):
    print('Running Riemannian GD...')
    x = [x0 for i in range(K)]
    t = [0 for i in range(K)]
    f = np.zeros((K,))
    f[0] = cost(x0)
    h = 1/L
    for k in range(K-1):
        t1 = time.time()
        x[k+1] = exp(x[k],-h*grad(x[k]))
        t2 = time.time()
        t[k+1] = t[k]+(t2-t1)
        f[k+1] = cost(x[k+1])
    return t, x, f

#This function is taken from "geometric_optimizers.py" in https://github.com/aorvieto/RAGDsDR (Alimisis et al., 2021)
#RAGD optimizer, by Zhang and Sra (2018) check http://proceedings.mlr.press/v75/zhang18a/zhang18a.pdf
def RAGD_optimizer(K,x0,L,mu,cost,grad,exp,log):
    print('Running Riemannian AGD (Zhang and Sra)...')
    x = [x0 for i in range(K)]
    t = [0 for i in range(K)]
    y = [0*x0 for i in range(K)]
    v = [x0 for i in range(K)]
    f = np.zeros((K,))
    a = np.zeros((K,))
    A = np.zeros((K,))
    f[0] = cost(x0)
    h = 1/L
    beta = np.sqrt(mu/L)/5
    alpha = (np.sqrt((beta**2)+4*(1+beta)*mu*h)-beta)/2
    gamma = mu*(np.sqrt(beta**2+4*(1+beta)*mu*h)-beta)/(np.sqrt(beta**2+4*(1+beta)*mu*h)+beta)
    gamma_bar = (1+beta)*gamma
    gradlist = [log(x0, x0) for i in range(K)]
    for k in range(K-1):
        t1 = time.time()
        y[k] = exp(x[k],(alpha*gamma/(gamma+alpha*mu))*log(x[k],v[k]))
        gradlist[k] = grad(y[k])
        x[k+1] = exp(y[k],-h*gradlist[k])
        v[k+1] = exp(y[k],(((1-alpha)*gamma)/gamma_bar)*log(y[k],v[k])-(alpha/gamma_bar)*gradlist[k])
        t2 = time.time()
        t[k+1] = t[k]+(t2-t1)
        f[k+1] = cost(x[k+1])
    return t, x, f

#This function is taken from "geometric_optimizers.py" in https://github.com/aorvieto/RAGDsDR (Alimisis et al., 2021) with modification
def RAGDsDR_optimizer(K,x0,L,zeta,cost,grad,exp,log,transp,line_search_iterations):
    if line_search_iterations>0:
        print('Running Riemannian AGDsDR(linesearch)...')
    else:
        print('Running Riemannian AGDsDR(no linesearch)...')
    x = [x0 for i in range(K)]
    t = [0 for i in range(K)]
    y = [0*x0 for i in range(K)]
    v = [x0 for i in range(K)]
    a = np.zeros((K,))
    beta = np.zeros((K,))
    A = np.zeros((K,))
    f = np.zeros((K,))
    f[0] = cost(x0)
    h = 1/L
    gradlist = [log(x0, x0) for i in range(K)]
    for k in range(K-1):
        t1 = time.time()
        beta[k] = k/(k+3)
        try:
            y[k] = exp(v[k], beta[k]* log(v[k],x[k]))
        except:
            print('An error occurred in the computation of y[k]!!, setting it to x[k]')
            y[k] = x[k]
        gradlist[k]=grad(y[k])
        x[k+1] = exp(y[k],-h*gradlist[k])
        a[k+1] = np.max(np.roots(np.array([zeta, -h, -h*A[k]])))
        A[k+1] = A[k]+a[k+1]
        try:
            v[k+1] = exp(v[k],-a[k+1]*transp(y[k],v[k],gradlist[k]))
        except:
            print('An error occurred in the computation of v[k+1]!!, setting it to v[k]')
            v[k+1] = v[k]
        t2 = time.time()
        t[k+1] = t[k]+(t2-t1)
        f[k+1] = cost(x[k+1])
    return t, x, f

# RNAG-SC in our paper
def RNAG_SC_optimizer(K, x0, L, mu, xi, cost, grad, exp, log, transp):
    print('Running RNAG-SC...')
    x = [x0 for i in range(K)]
    t = [0 for i in range(K)]
    y = [0*x0 for i in range(K)]
    v = [log(x0, x0) for i in range(K)]
    f = np.zeros((K,))
    f[0] = cost(x0)
    h = 1/L
    q = mu/L
    gradlist = [log(x0, x0) for i in range(K)]
    for k in range(K-1):
        t1 = time.time()
        try:
            y[k] = exp(x[k], ((q**(1/2)) / ((1 / (xi ** 0.5)) + q ** (1 / 2))) * v[k])
        except:
            y[k] = x[k]
        try:
            gradlist[k]=grad(y[k])
            v[k] = transp(x[k], y[k], (1 - ((q ** (1 / 2)) / ((1 / (xi ** 0.5)) + q ** (1 / 2)))) * v[k])
            x[k + 1] = exp(y[k], -h * gradlist[k])
            v[k + 1] = (1 - (q / xi) ** (1 / 2)) * v[k] + ((q / xi) ** (1 / 2)) * (
                        -(1 / mu) * gradlist[k])
            v[k + 1] = transp(y[k], x[k + 1], v[k + 1] - (-h * gradlist[k]))
        except:
            print('An error occurred in the computation of y[k]!!, do vanilla gradient descent step')
            x[k + 1] = exp(x[k], -h * grad(x[k]))
            v[k + 1] = log(x[k + 1], x[k + 1])
        t2 = time.time()
        t[k+1] = t[k]+(t2-t1)
        f[k+1] = cost(x[k+1])
    return t, x, f

# RNAG-C in our paper
def RNAG_C_optimizer(K, x0, L, xi, cost, grad, exp, log, transp):
    print('Running RNAG-C...')
    x = [x0 for i in range(K)]
    t = [0 for i in range(K)]
    y = [0*x0 for i in range(K)]
    v = [log(x0, x0) for i in range(K)]
    f = np.zeros((K,))
    f[0] = cost(x0)
    h = 1/L
    gradlist = [log(x0, x0) for i in range(K)]
    for k in range(K-1):
        t1 = time.time()
        lmbda = (k + 2 * xi) / 2
        try:
            y[k] = exp(x[k], (xi / (lmbda + xi - 1)) * v[k])
        except:
            y[k] = x[k]
        try:
            gradlist[k] = grad(y[k])
            v[k] = transp(x[k], y[k], v[k] - log(x[k],y[k]))
            x[k + 1] = exp(y[k], -h * gradlist[k])
            v[k + 1] = v[k] + (-h * lmbda / xi) * gradlist[k]
            v[k + 1] = transp(y[k], x[k + 1], v[k + 1] - (-h * gradlist[k]))
        except:
            print('An error occurred in the computation of y[k]!!, do vanilla gradient descent step')
            x[k + 1] = exp(x[k], -h * grad(x[k]))
            v[k + 1] = log(x[k + 1], x[k + 1])
        t2 = time.time()
        t[k+1] = t[k]+(t2-t1)
        f[k+1] = cost(x[k+1])
    return t, x, f

# SIRNAG for geodesically strongly convex case (Alimisis et al., 2020)
def SIRNAG_SC_optimizer(K,x0,h,mu,zeta,cost,grad,exp,log,transp):
    print('Running SIRNAG-SC (Alimisis et al., 2020)...')
    x = [x0 for i in range(K)]
    t = [0 for i in range(K)]
    a = [log(x0, x0) for i in range(K)]
    v = [log(x0, x0) for i in range(K)]
    f = np.zeros((K,))
    f[0] = cost(x0)
    beta = 1-(zeta**0.5 + 1/(zeta**0.5))*h*(mu**0.5)
    for k in range(K - 1):
        t1 = time.time()
        a[k] = beta*v[k] - h*grad(x[k])
        x[k+1] = exp(x[k],h*a[k])
        v[k+1] = transp(x[k],x[k+1],a[k])
        f[k+1] = cost(x[k+1])
        t2 = time.time()
        t[k+1] = t[k]+(t2-t1)
        f[k+1] = cost(x[k+1])
    return t, x, f

# SIRNAG for geodesically convex case (Alimisis et al., 2020)
def SIRNAG_C_optimizer(K,x0,h,zeta,cost,grad,exp,log,transp):
    print('Running SIRNAG-C (Alimisis et al., 2020)...')
    x = [x0 for i in range(K)]
    t = [0 for i in range(K)]
    a = [log(x0, x0) for i in range(K)]
    v = [log(x0, x0) for i in range(K)]
    f = np.zeros((K,))
    f[0] = cost(x0)
    beta = 1
    for k in range(K - 1):
        beta = (k)/(k+1+2*zeta)
        t1 = time.time()
        a[k] = beta*v[k] - h*grad(x[k])
        x[k+1] = exp(x[k],h*a[k])
        v[k+1] = transp(x[k],x[k+1],a[k])
        f[k+1] = cost(x[k+1])
        t2 = time.time()
        t[k+1] = t[k]+(t2-t1)
        f[k+1] = cost(x[k+1])
    return t, x, f