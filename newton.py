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

N=100
dim=3*N+2
barrier_locs=[math.floor(N/3),2*math.floor(N/3)]
pdim=barrier_locs[1]-barrier_locs[0] # number of parameters

dx=2*10**-9 # 1 nm
melec=0.18*9.1*10**(-31) # effective mass of electron
mhole=1.4*9.1*10**(-31) # effective mass of hole
hbar=1.0545*10**(-34) # reduced Planck constant
epsilon=8.854*10**(-12)
qelec=-1.602*10**(-19) # -1.602*10**(-19)
qhole=-qelec
sheet_charge=10**-12
evperj=6.2415091*10**(18) # eV per J
threshold1=10**-12 # convergence thresholds
threshold2=10**-12 # convergence thresholds
threshold3=10**-8 # convergence thresholds

depth=5.2*10**-23
vcb=np.concatenate((np.linspace(0,depth/10,math.floor(N/3)),np.linspace(0,-depth,math.floor(N/3)),np.linspace(-depth+depth/10,-depth+2*depth/10,N-2*math.floor(N/3))))
vvb=np.flip(vcb,axis=0)

"""
def bands(d,x,barrier_locs,surface_sigma):
    #""
    #Returns two numbers, V(d) (in eV) for the conduction and valence bands,
    #respectively, as a function of distance d along the transport dimension
    #and indium mole fraction x within the doped region.
    #""
    b=np.array([0.,0.])
    if d>=barrier_locs[0] and d<barrier_locs[1]:
        #""
        #Indium doped region.
        #""
        b[0]=-(x*5.6+(1-x)*4.1-1.43*x*(1-x))
        b[1]=-(x*5.6+(1-x)*4.1-1.43*x*(1-x))-(x*0.7+(1-x)*3.42-1.43*x*(1-x))
    else:
        #""
        #Regular gallium nitride.
        #""
        b[0]=-4.1
        b[1]=-4.1-3.42
    if d>=barrier_locs[0] and d<barrier_locs[1]:
        #""
        #Surface charge density causes a discontinuity in the slope of V.
        #""
        b[0]-=(d-barrier_locs[0])*(surface_sigma/epsilon)/dx
        b[1]-=(d-barrier_locs[0])*(surface_sigma/epsilon)/dx
    elif d<barrier_locs[0]:
        b[0]+=(d-barrier_locs[0])*(surface_sigma/epsilon)/dx
        b[1]+=(d-barrier_locs[0])*(surface_sigma/epsilon)/dx
    else:
        b[0]+=(d-barrier_locs[1])*(surface_sigma/epsilon)/dx
        b[1]+=(d-barrier_locs[1])*(surface_sigma/epsilon)/dx
    return b/evperj
"""

def rho(v):
    """
    Takes as input a state vector v
    and outputs the volumetric charge density of the system.
    Rho depends on psi so there are now derivatives that must
    be accounted for in the Poisson portion of the Jacobian.
    """
    s=copy.copy(v)
    cd=np.zeros(N)
    for i in range(N):
        if i==barrier_locs[0]:
            cd[i]=(sheet_charge/dx)+qelec*(abs(s[i]))**2 + qhole*(abs(s[i+N]))**2
        elif i==barrier_locs[1]:
            cd[i]=(-sheet_charge/dx)+qelec*(abs(s[i]))**2 + qhole*(abs(s[i+N]))**2
        else:
            cd[i]=qelec*(abs(s[i]))**2 + qhole*(abs(s[i+N]))**2
    return cd

def Jacobian(v,params,vcb,vvb):
    """
    Takes as input a state vector s and a parameter vector p
    and outputs the Jacobian of the system evaluated at s.
    Rho is independent of p but depends on psi, while the Schrodinger
    equations now depend on p.
    """
    s=copy.copy(v)
    p=copy.copy(params)
    jacob=np.zeros((dim,dim))
    s[0:N]/=np.linalg.norm(s[0:N])
    s[N:2*N]/=np.linalg.norm(s[N:2*N])
    for i in range(dim):
        for j in range(dim):
            if i<N:
                """
                Electron equations.
                """
                if i==j:
                    if i>=barrier_locs[0] and i<barrier_locs[1]:
                        jacob[i,j]=-(2+((2*melec*dx**2)/(hbar**2))*(vcb[i]+p[i-barrier_locs[0]]+qelec*s[i+2*N])-((2*melec*dx**2)/(hbar**2))*s[-2])
                    else:
                        jacob[i,j]=-(2+((2*melec*dx**2)/(hbar**2))*(vcb[i]+qelec*s[i+2*N])-((2*melec*dx**2)/(hbar**2))*s[-2])
                if i==(j+1)%N and j<N:
                    jacob[i,j]=1
                if i==(j-1)%N and j<N:
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
                    if i-N>=barrier_locs[0] and i-N<barrier_locs[1]:
                        jacob[i,j]=-(2+((2*mhole*dx**2)/(hbar**2))*(vvb[i-N]-p[i-N-barrier_locs[0]]+qhole*s[i+N])-((2*mhole*dx**2)/(hbar**2))*s[-1])
                    else:
                        jacob[i,j]=-(2+((2*mhole*dx**2)/(hbar**2))*(vvb[i-N]+qhole*s[i+N])-((2*mhole*dx**2)/(hbar**2))*s[-1])
                if i-N==(j+1-N)%N and j<2*N and j>=N:
                    jacob[i,j]=1
                if i-N==(j-1-N)%N and j<2*N and j>=N:
                    jacob[i,j]=1
                if j==i+N:
                    jacob[i,j]=-((2*mhole*dx**2)/(hbar**2))*qhole*s[i]
                if j==(dim-1):
                    jacob[i,j]=((2*mhole*dx**2)/(hbar**2))*s[i]
            if i>=2*N and i<3*N:
                """
                Poisson equations.
                """
                if i-2*N==j: # Rho depends on psi so there are nonzero derivatives
                    jacob[i,j]=2*qelec*s[j]
                if i-N==j:
                    jacob[i,j]=2*qhole*s[j]
                if i==j:
                    jacob[i,j]=-2
                if i-2*N==(j-1-2*N)%N and j<3*N and j>=2*N:
                    jacob[i,j]=1
                if i-2*N==(j+1-2*N)%N and j<3*N and j>=2*N:
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
                    
def RHS(v,params,vcb,vvb):
    """
    Takes as input (3*N+2)-dimensional a state vector s
    and an 2*N-dimensional parameter vector p
    and outputs the RHS of the system, which is -f(s,p).
    N.B. that this minus is *included* in the RHS definition.
    """
    s=copy.copy(v)
    p=copy.copy(params)
    rv=rho(s)
    #s[0:N]/=np.linalg.norm(s[0:N])
    #s[N:2*N]/=np.linalg.norm(s[N:2*N])
    rhside=np.zeros(dim)
    for i in range(dim):
        if i<N:
            if i>=barrier_locs[0] and i<barrier_locs[1]:
                rhside[i]=s[(i-1)%N]+s[(i+1)%N]-(2+((2*melec*dx**2)/(hbar**2))*(vcb[i]+p[i-barrier_locs[0]]+qelec*s[i+2*N])-((2*melec*dx**2)/(hbar**2))*s[-2])*s[i]
            else:
                rhside[i]=s[(i-1)%N]+s[(i+1)%N]-(2+((2*melec*dx**2)/(hbar**2))*(vcb[i]+qelec*s[i+2*N])-((2*melec*dx**2)/(hbar**2))*s[-2])*s[i]
        if i>=N and i<2*N:
            if i-N>=barrier_locs[0] and i-N<barrier_locs[1]:
                rhside[i]=s[(i-1)%N+N]+s[(i+1)%N+N]-(2+((2*mhole*dx**2)/(hbar**2))*(vvb[i-N]-p[i-N-barrier_locs[0]]+qhole*s[i+N])-((2*mhole*dx**2)/(hbar**2))*s[-1])*s[i]
            else:
                rhside[i]=s[(i-1)%N+N]+s[(i+1)%N+N]-(2+((2*mhole*dx**2)/(hbar**2))*(vvb[i-N]+qhole*s[i+N])-((2*mhole*dx**2)/(hbar**2))*s[-1])*s[i]
        if i>=2*N and i<3*N:
            rhside[i]=s[(i-1)%N+2*N]+s[(i+1)%N+2*N]-2*s[i]+(dx**2)*(rv[i-2*N])/epsilon
        if i==3*N:
            rhside[i]=np.sum(np.square(s[0:N]))-1
        if i==3*N+1:
            rhside[i]=np.sum(np.square(s[N:2*N]))-1
    np.savetxt("rhs.csv",rhside,delimiter=',',fmt='%.6f')
    return -1*rhside
        

def solve(guess,params,vcb,vvb,quiet=False,save=False):
    """
    Applies Newton's method to the system defined by the Jacobian and a given vector of parameters.
    In this context, the vector of parameters is the material composition.
    """
    solstart=datetime.datetime.now()
    s=copy.copy(guess)
    p=copy.copy(params)
    #s[0:N]/=np.linalg.norm(s[0:N])
    #s[N:2*N]/=np.linalg.norm(s[N:2*N])
    print("rhsnorms:",np.linalg.norm(RHS(s,p,vcb,vvb)[0:N]),np.linalg.norm(RHS(s,p,vcb,vvb)[N:2*N]),np.linalg.norm(RHS(s,p,vcb,vvb)[2*N:3*N]))
    while np.linalg.norm(RHS(s,p,vcb,vvb)[0:N])>threshold1 or np.linalg.norm(RHS(s,p,vcb,vvb)[N:2*N])>threshold2 or np.linalg.norm(RHS(s,p,vcb,vvb)[2*N:3*N])>threshold3:
        ds=np.linalg.solve(Jacobian(s,p,vcb,vvb),RHS(s,p,vcb,vvb))
        s+=ds
        s[0:N]/=np.linalg.norm(s[0:N])
        s[N:2*N]/=np.linalg.norm(s[N:2*N])
        if quiet is False:
            print("rhsnorms:",np.linalg.norm(RHS(s,p,vcb,vvb)[0:N]),np.linalg.norm(RHS(s,p,vcb,vvb)[N:2*N]),np.linalg.norm(RHS(s,p,vcb,vvb)[2*N:3*N]))
            #print("electron energy (eV): ",s[-2]*evperj)
            #print("hole energy (eV): ",s[-1]*evperj)
    np.savetxt("solution.csv",s,delimiter=',',fmt='%.6f')
    plt.figure(1,dpi=120)
    plt.plot(np.square(s[0:N]))
    plt.plot(np.square(s[N:2*N]))
    plt.show()
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

def solveadjoint(xsol,params,vcb,vvb):
    """
    Returns the lambda that solves the adjoint problem.
    Note that the adjoint problem is a single linear equation;
    it does not need Newton iteration.
    """
    s=copy.copy(xsol)
    p=copy.copy(params)
    adjsol=np.linalg.solve(Jacobian(s,p,vcb,vvb).T,dgdx(s))
    return adjsol

def dgdp(xsol,params,vcb,vvb):
    """
    Returns the gradient of g w.r.t the parameter p
    evaluated at a solution xsol of f s.t. f(xsol)=0.
    dfdp is an M by P matrix where M is the number of equations,
    in this case M=dim=3*N+2. Each entry dfdp_i,j is the
    derivative of the ith equation w.r.t. the jth parameter.
    """
    s=copy.copy(xsol)
    p=copy.copy(params)
    dfdp=np.zeros((dim,pdim))
    for i in range(dim):
        for j in range(pdim):
            if i<N and i==j+barrier_locs[0]:
                if i>=barrier_locs[0] and i<barrier_locs[1]:
                    dfdp[i,j]=-((2*melec*dx**2)/(hbar**2))*s[i]
            if i>=N and i<2*N and i==j+barrier_locs[0]:
                if i-N>=barrier_locs[0] and i-N<barrier_locs[1]:
                    dfdp[i,j]=((2*mhole*dx**2)/(hbar**2))*s[i]
    return -np.matmul(solveadjoint(s,p,vcb,vvb),dfdp)

def ascent(xsol,params,vcb,vvb,alpha=1,max=50,savefig=False):
    #Applies gradient ascent to find optimal parameters using a learning rate alpha.
    s=copy.copy(xsol)
    p=copy.copy(params)
    oldg=0.
    newg=1.
    cacheg=0
    cachep=np.zeros(len(p))
    gs=[]
    for i in range(max):
        print("iteration",i+1,"of",max)
        oldg=g(s)
        grad=dgdp(s,p,vcb,vvb)
        p=p+alpha*grad
        s=solve(s,p,vcb,vvb,quiet=False) # solve(np.concatenate((np.full(N,1/math.sqrt(N)),np.full(N,1/math.sqrt(N)),np.ones(N),(10**-15)*np.ones(2))),p,quiet=True) # solve(s,p,quiet=True)
        newg=g(s)
        gs.append(newg)
        print("newg:",newg)
        print("oldg:",oldg)
        print("difference:",newg-oldg)
        dv=np.concatenate((np.zeros(barrier_locs[0]+1),p,np.zeros(N-barrier_locs[1]-1)))
        if newg>cacheg:
            cacheg=copy.copy(newg)
            cachep=copy.deepcopy(p)
        print("\n")
        if savefig is True:
            plt.figure(1,dpi=120)
            plt.plot(-vvb+dv)
            plt.xlabel("Lattice label")
            plt.ylabel("Valence band (J)")
            plt.title("Overlap: "+'%.10f'%newg)
            plt.savefig('images/'+str(i)+'.png',dpi=300)
    #plt.figure(1)
    #plt.xlabel("Lattice label")
    #plt.ylabel("Probability density")
    #plt.show()
    print(gs)
    return cachep

def gaussian(x, mu, sig):
    return (1/np.sqrt(2*np.pi*np.power(sig, 2.)))*np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

start=datetime.datetime.now()

initial_guess=np.concatenate((np.array([gaussian(x,80,130) for x in range(N)]),np.array([gaussian(x,20,50) for x in range(N)]),np.ones(N),np.array([10**-19,10**-19]))) # gaussian(np.linspace(0,N,N),math.floor(N/2),10)

#initial_guess=np.concatenate((0.07*np.sin(np.pi*np.linspace(0,N,N)/N),0.07*np.sin(np.pi*np.linspace(0,N,N)/N),np.ones(N),np.array([10**-19,10**-19]))) # np.full(N,1/math.sqrt(N))
#initial_guess=np.concatenate((np.full(N,1/math.sqrt(N)),np.full(N,1/math.sqrt(N)),1*np.ones(N),np.array([10**-19,10**-19]))) # np.full(N,1/math.sqrt(N))
#initial_guess=np.random.random(dim)

plt.plot(vcb)
plt.plot(vvb)
plt.show()

initial_params=np.zeros(pdim)

sol=solve(initial_guess,initial_params,vcb,vvb)
grad=dgdp(sol,initial_params,vcb,vvb)
print("g(sol):",g(sol))
print("energies (eV):",sol[-2]*evperj,sol[-1]*evperj)

optimalp=ascent(sol,initial_params,vcb,vvb,alpha=1.5*10**-45,max=500,savefig=True)
optimaldv=np.concatenate((np.zeros(barrier_locs[0]),optimalp,np.zeros(N-barrier_locs[1])))
vcb_op=vcb+optimaldv
vvb_op=vvb+optimaldv

print("total runtime",datetime.datetime.now()-start)
