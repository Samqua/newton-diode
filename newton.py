# -*- coding: utf-8 -*-
"""
Created on Tue Aug  7 17:32:39 2018

@author: Samqua
"""

import datetime
import numpy as np
import copy
import math
import matplotlib.pyplot as plt
#import random

N=400
dim=3*N+2

dx=5*10**(-10) # spacing between lattice points
#dx=(1*10**(-8))/N
melec=0.18*9.1*10**(-31) # effective mass of electron
mhole=1.4*9.1*10**(-31) # effective mass of hole
hbar=1.0545*10**(-34) # reduced Planck constant
epsilon=1
qelec=-1.602*10**(-19) # -1.602*10**(-19)
qhole=-qelec
evperj=6.2415091*10**(18) # eV per J
#dx=0.01
#melec=0.18
#mhole=1.4
#hbar=1.
threshold1=10**-11 # convergence thresholds
threshold2=10**-11 # convergence thresholds
threshold3=10**-11 # convergence thresholds

def Jacobian(v):
    """
    Takes as input a state vector v
    and outputs the Jacobian of the system evaluated at v.
    """
    s=copy.copy(v)
    jacob=np.zeros((dim,dim))
    #s[0:N]/=np.linalg.norm(s[0:N])
    #s[N:2*N]/=np.linalg.norm(s[N:2*N])
    for i in range(dim):
        for j in range(dim):
            if i<N:
                """
                Electron equations.
                """
                if i==j:
                    jacob[i,j]=-(2+((2*melec*dx**2)/(hbar**2))*qelec*s[i+2*N]-((2*melec*dx**2)/(hbar**2))*s[-2])
                if i==(j+1) and j<N:
                    jacob[i,j]=1
                if i==(j-1) and j<N:
                    jacob[i,j]=1
                if j==i+2*N:
                    jacob[i,j]=-((2*melec*dx**2)/(hbar**2))*qelec*s[i]
                if j==(dim-2):
                    jacob[i,j]=((2*melec*dx**2)/(hbar**2))*s[i]
            if i>=N and i<2*N:
                """
                Hole equations.
                """
                if i==j:
                    jacob[i,j]=-(2+((2*mhole*dx**2)/(hbar**2))*qhole*s[i+N]-((2*mhole*dx**2)/(hbar**2))*s[-1])
                if i==(j+1) and j<2*N and j>=N:
                    jacob[i,j]=1
                if i==(j-1) and j<2*N and j>=N:
                    jacob[i,j]=1
                if j==i+N:
                    jacob[i,j]=-((2*mhole*dx**2)/(hbar**2))*qhole*s[i]
                if j==(dim-1):
                    jacob[i,j]=((2*mhole*dx**2)/(hbar**2))*s[i]
            if i>=2*N and i<3*N:
                """
                Poisson equations.
                """
                if i==j:
                    jacob[i,j]=-2
                if i==(j-1) and j<3*N and j>=2*N:
                    jacob[i,j]=1
                if i==(j+1) and j<3*N and j>=2*N:
                    jacob[i,j]=1
            if i==3*N:
                """
                Electron wavefunction normalization.
                """
                if j<N:
                    jacob[i,j]=2*s[j]
            if i==3*N+1:
                """
                Hole wavefunction normalization.
                """
                if j>=N and j<2*N:
                    jacob[i,j]=2*s[j]
    np.savetxt("jacobian.csv",jacob,delimiter=',',fmt='%.6f')
    return jacob
                    
def RHS(v,params):
    """
    Takes as input (3*N+2)-dimensional a state vector s
    and an N-dimensional parameter vector p
    and outputs the RHS of the system, which is -f(s,p).
    N.B. that this minus is *included* in the RHS definition.
    """
    s=copy.copy(v)
    p=copy.copy(params)
    #s[0:N]/=np.linalg.norm(s[0:N])
    #s[N:2*N]/=np.linalg.norm(s[N:2*N])
    rhside=np.zeros(dim)
    for i in range(dim):
        if i<N:
            if i==0:
                rhside[i]=s[i+1]-(2+((2*melec*dx**2)/(hbar**2))*qelec*s[i+2*N]-((2*melec*dx**2)/(hbar**2))*s[-2])*s[i]
            elif i==N-1:
                rhside[i]=s[i-1]-(2+((2*melec*dx**2)/(hbar**2))*qelec*s[i+2*N]-((2*melec*dx**2)/(hbar**2))*s[-2])*s[i]
            else:
                rhside[i]=s[i-1]+s[i+1]-(2+((2*melec*dx**2)/(hbar**2))*qelec*s[i+2*N]-((2*melec*dx**2)/(hbar**2))*s[-2])*s[i]
        if i>=N and i<2*N:
            if i==N:
                rhside[i]=s[i+1]-(2+((2*mhole*dx**2)/(hbar**2))*qhole*s[i+N]-((2*mhole*dx**2)/(hbar**2))*s[-1])*s[i]
            elif i==2*N-1:
                rhside[i]=s[i-1]-(2+((2*mhole*dx**2)/(hbar**2))*qhole*s[i+N]-((2*mhole*dx**2)/(hbar**2))*s[-1])*s[i]
            else:
                rhside[i]=s[i-1]+s[i+1]-(2+((2*mhole*dx**2)/(hbar**2))*qhole*s[i+N]-((2*mhole*dx**2)/(hbar**2))*s[-1])*s[i]
        if i>=2*N and i<3*N:
            if i==2*N:
                rhside[i]=s[i+1]-2*s[i]+(dx**2)*p[i-2*N]/epsilon
            elif i==3*N-1:
                rhside[i]=s[i-1]-2*s[i]+(dx**2)*p[i-2*N]/epsilon
            else:
                rhside[i]=s[i-1]+s[i+1]-2*s[i]+(dx**2)*p[i-2*N]/epsilon
        if i==3*N:
            rhside[i]=np.sum(np.square(s[0:N]))-1
        if i==3*N+1:
            rhside[i]=np.sum(np.square(s[N:2*N]))-1
    np.savetxt("rhs.csv",rhside,delimiter=',',fmt='%.6f')
    return -1*rhside
        

def solve(guess,params,quiet=False,save=False):
    """
    Applies Newton's method to the system defined by the Jacobian and a given vector of parameters.
    In this context, the vector of parameters is the material composition.
    """
    solstart=datetime.datetime.now()
    s=copy.copy(guess)
    p=copy.copy(params)
    #s[0:N]/=np.linalg.norm(s[0:N])
    #s[N:2*N]/=np.linalg.norm(s[N:2*N])
    print("rhsnorms:",np.linalg.norm(RHS(s,p)[0:N]),np.linalg.norm(RHS(s,p)[N:2*N]),np.linalg.norm(RHS(s,p)[2*N:3*N]))
    while np.linalg.norm(RHS(s,p)[0:N])>threshold1 or np.linalg.norm(RHS(s,p)[N:2*N])>threshold2 or np.linalg.norm(RHS(s,p)[2*N:3*N])>threshold3:
        ds=np.linalg.solve(Jacobian(s),RHS(s,p))
        s+=ds
        s[0:N]/=np.linalg.norm(s[0:N])
        s[N:2*N]/=np.linalg.norm(s[N:2*N])
        if quiet is False:
            print("rhsnorms:",np.linalg.norm(RHS(s,p)[0:N]),np.linalg.norm(RHS(s,p)[N:2*N]),np.linalg.norm(RHS(s,p)[2*N:3*N]))
            #print("electron energy (eV): ",s[-2]*evperj)
            #print("hole energy (eV): ",s[-1]*evperj)
    np.savetxt("solution.csv",s,delimiter=',',fmt='%.6f')
    plt.figure(1,dpi=120)
    plt.plot(np.square(s[0:N]))
    plt.plot(np.square(s[N:2*N]))
    plt.show()
    plt.clf()
    if save is True:
        plt.savefig('images/'+str(np.linalg.norm(RHS(s,p)))+'.png',dpi=300)
        plt.clf()
    print("solve runtime:",datetime.datetime.now()-solstart)
    return s

def g(v):
    """
    Returns the function of interest,
    abs(sum(psi_e,i * psi_h*i) over lattice points i)
    """
    s=copy.copy(v)
    return abs(np.sum(s[0:N]*s[N:2*N]))

def dgdx(v):
    """
    Partial derivative of g w.r.t. x.
    """
    s=copy.copy(v)
    out=np.zeros(dim)
    for i in range(0,N):
        out[i]=(s[i+N]*np.sum(s[0:N]*s[N:2*N]))/(g(s))
    for i in range(N,2*N):
        out[i]=(s[i-N]*np.sum(s[0:N]*s[N:2*N]))/(g(s))
    return out

def solveadjoint(xsol):
    """
    Returns the lambda that solves the adjoint problem.
    Note that the adjoint problem is a single linear equation;
    it does not need Newton iteration.
    """
    s=copy.copy(xsol)
    adjsol=np.linalg.solve(Jacobian(s).T,dgdx(s))
    return adjsol

def dgdp(xsol):
    """
    Returns the gradient of g w.r.t the parameter p
    evaluated at a solution xsol of f s.t. f(xsol)=0.
    dfdp is an M x P matrix where M is the number of equations,
    in this case M=dim=3*N+2. Each entry dfdp_i,j is the
    derivative of the ith equation w.r.t. the jth parameter.
    It's all zeros except a diagonal portion from i=2*N to i=3*N.
    """
    s=copy.copy(xsol)
    dfdp=np.zeros((dim,N))
    for i in range(2*N,3*N):
        for j in range(N):
            if (i-2*N)==j:
                dfdp[i,j]=(dx**2)/epsilon
    return -np.matmul(solveadjoint(s),dfdp)

def ascent(xsol,params,alpha=1,max=50):
    """
    Applies gradient ascent to find optimal parameters
    using a learning rate alpha.
    """
    s=copy.copy(xsol)
    p=copy.copy(params)
    oldg=0.
    newg=1.
    cacheg=0
    cachep=np.zeros(len(p))
    for i in range(max):
        print("iteration",i+1,"of",max)
        oldg=g(s)
        grad=dgdp(s)
        p=p+alpha*grad
        s=solve(np.concatenate((np.full(N,1/math.sqrt(N)),np.full(N,1/math.sqrt(N)),np.ones(N),np.ones(2))),p,quiet=False) # solve(np.concatenate((np.full(N,1/math.sqrt(N)),np.full(N,1/math.sqrt(N)),np.ones(N),(10**-15)*np.ones(2))),p,quiet=True) # solve(s,p,quiet=True)
        newg=g(s)
        print("newg:",newg)
        print("oldg:",oldg)
        print("difference:",newg-oldg)
        if newg>cacheg:
            cacheg=copy.copy(newg)
            cachep=copy.deepcopy(p)
        print("\n")
    #plt.figure(1)
    #plt.xlabel("Lattice label")
    #plt.ylabel("Probability density")
    #plt.show()
    return cachep

initial_guess=np.concatenate((np.full(N,1/math.sqrt(N)),np.full(N,1/math.sqrt(N)),1*np.ones(N),np.ones(2)))
#initial_guess=np.random.random(size=dim)
initial_params=np.concatenate((-10**3*np.ones(math.floor(N/2)),10**3*np.ones(math.ceil(N/2))))

t0=datetime.datetime.now()

sol=solve(initial_guess,initial_params)
grad=dgdp(sol)
plt.clf()
plt.plot(grad)
print("electron energy:",sol[-2])
print("hole energy:",sol[-1])
#print("electron energy (eV): ",sol[-2]*evperj)
#print("hole energy (eV): ",sol[-1]*evperj)
#print("gap (eV):",(sol[-2]-sol[-1])*evperj)
print("overlap functional g(sol): ",g(sol))

optimalp=ascent(sol,initial_params,alpha=3*10**27,max=10)
plt.clf()
plt.plot(optimalp)
plt.show()
print("total runtime: ",datetime.datetime.now()-t0)