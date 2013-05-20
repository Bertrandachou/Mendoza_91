from __future__ import division

# -*- coding: utf-8 -*-
"""
Created on Tue May 07 17:58:10 2013

@author: Bertrand Achou
"""

""" This is an attempt to reproduce the results obtained by Mendoza
in his paper 'RBC in a small open economy' published in the AER in 1991.
We use vectorization in order to perform the task faster"""


import numpy as np
import time


# First, we define our grid of parameters

p = {'alpha': 0.32, 'rstar': 0.04, 'gamma': 1.001 , 'delta': 0.1, 'omega': 1.455, 'beta': 0.11, 'phi': 0, 'epsilon': 10**(-6) }

# Then the grids for the stochastic process (transition and values)

p_stoch = {'ep': 1.18/100, 'n': 0, 'rho': 0, 'rho_en': 0.34}

# We define the parameters of the grids

pgrid = {'kmin': 3.25, 'kmax': 3.56 , 'nk': 22, 'Amin': -1.42 , 'Amax': 0.08 , 'nA': 22 }

 
# We build the transition matrix that will be used

Pi          = (p_stoch['rho'] + 1) / 4

stoch_transit = np.array([\
[(1-p_stoch['rho_en']) * Pi + p_stoch['rho_en'], (1-p_stoch['rho_en']) * (0.5 - Pi), (1-p_stoch['rho_en']) * (0.5 - Pi), (1-p_stoch['rho_en']) * Pi],\
[(1-p_stoch['rho_en']) * Pi, (1-p_stoch['rho_en']) * (0.5 - Pi) + p_stoch['rho_en'] , (1-p_stoch['rho_en']) * (0.5 - Pi), (1-p_stoch['rho_en']) * Pi],\
[(1-p_stoch['rho_en']) * Pi, (1-p_stoch['rho_en']) * (0.5 - Pi), (1-p_stoch['rho_en']) * (0.5 - Pi) + p_stoch['rho_en'], (1-p_stoch['rho_en']) * Pi],\
[(1-p_stoch['rho_en']) * Pi, (1-p_stoch['rho_en']) * (0.5 - Pi), (1-p_stoch['rho_en']) * (0.5 - Pi), (1-p_stoch['rho_en']) * Pi + p_stoch['rho_en']]\
])


# Given these parameters we build our grids respectively for
# k, kprime, A, Aprime, the stochastic states (e and n) 


# the grid for K

klin =  np.array([np.linspace(pgrid['kmin'],pgrid['kmax'],pgrid['nk'])])
kgrid_temp = np.repeat( klin , 4 * pgrid['nA'] * pgrid['nA'] * pgrid['nk'], 0 )
kgrid      = np.reshape(kgrid_temp, (4 , pgrid['nA'], pgrid['nA'] , pgrid['nk'] , pgrid['nk']) )

# the grid for K'

kpgrid_temp = np.repeat(np.repeat(klin, pgrid['nk'], 1), 4 * pgrid['nA'] * pgrid['nA'], 0)
kpgrid      = np.reshape(kpgrid_temp, (4 , pgrid['nA'], pgrid['nA'] , pgrid['nk'] , pgrid['nk']) )

# the grid for A

Alin  = np.linspace(pgrid['Amin'],pgrid['Amax'],pgrid['nA'])
Agrid_temp1 = np.array([np.repeat( Alin , (pgrid['nk']**2) * pgrid['nA'] , 0 )])
Agrid_temp2 = np.repeat( Agrid_temp1 , 4 , 0 )
Agrid       = np.reshape( Agrid_temp2 , (4 , pgrid['nA'], pgrid['nA'] , pgrid['nk'] , pgrid['nk']) )

# the grid for A'

Apgrid_temp1 = np.array([np.repeat( Alin , (pgrid['nk']**2)  , 0 )])
Apgrid_temp2 = np.array([np.repeat( Apgrid_temp1 , pgrid['nA'] , 0 )])
Apgrid_temp3 = np.repeat( Apgrid_temp2 , 4 , 0 )
Apgrid       = np.reshape( Apgrid_temp3 , (4 , pgrid['nA'], pgrid['nA'] , pgrid['nk'] , pgrid['nk']) )

# the grid for e (the level of productivity)

egrid = np.zeros ( (4 , pgrid['nA'], pgrid['nA'] , pgrid['nk'] , pgrid['nk']) )
egrid[0:2].fill(p_stoch['ep'])
egrid[2:].fill(- p_stoch['ep'])

# the grid for n (which affects the return on foreign assets)

ngrid = np.zeros ( (4 , pgrid['nA'], pgrid['nA'] , pgrid['nk'] , pgrid['nk']) )
ngrid[0].fill(p_stoch['n'])
ngrid[1].fill(-p_stoch['n'])
ngrid[2].fill(p_stoch['n'])
ngrid[3].fill(-p_stoch['n'])



# now we define our functions 

import numexpr

# a function that computes the optimal labor as a function of k,e and the parameters p

def labour(k,eg,p):
    alpha = p['alpha']
    omega = p['omega']
    return numexpr.evaluate( '( (1 - alpha) * exp(eg) * (k**alpha) ) ** (1 / ( alpha + omega - 1))' )
    

def composite_consumption(c,l,p):
    omega = p['omega']
    return numexpr.evaluate( 'c - (l**omega)/omega' )
    
# utility as a function of c, l and the parameters p

def utility(cc,p):
    gamma = p['gamma']
    if gamma == 1:
        return numexpr.evaluate( 'log(cc)' )
    else:
        return numexpr.evaluate( '( (cc)**(1-gamma) - 1 ) / (1 - gamma)' )
    
# a function which returns the discount factor from c, l, e and the parameters p

def discount(cc,p):
    beta = p['beta']
    return numexpr.evaluate( 'exp(-beta * log(1 + cc ))' )
    
    
# production function as a function k, k', l, e and the parameters p

def production(k,kp,l,eg,p):
    alpha = p['alpha']
    phi   = p['phi']
    return  numexpr.evaluate( 'exp(eg) * (k**alpha) * (l**(1-alpha)) - (phi/2) * (kp - k)**2' )
    

# we define our grid for value function
# first dimension is the stochastic state dimension
# second is the foreign asset dimension (the A dimension)
# third is the domestic capital dimension (the k dimension)

V0t = np.zeros((4 , pgrid['nA'] , pgrid['nk']))
V0  = V0t.reshape( (4 , pgrid['nA'] * pgrid['nk'] ) )




# we need to build a function which given our matrices, parameters and initial value function
# returns the new value function from the recursive algorithm
# this is the main part of the program
# it depends on k, k', A, A', the matrix of e and n defined here as stoch
# on p the grid of parameters
# on the transition matrix of the stochastic process here called transit

def bud(k,kp,A,Ap,eg,n,p):
    
    # we first need to compute the instantaneouse utility of the agent for every
    # state and possible decision he can take
    # when consumption is negative we put it to 0.001 and assigne a large negative utility
    # so that it is never optimal for the agent to choose these values
    delt = p['delta']
    omeg = p['omega']
    rs   = p['rstar']
    
    ltemp = labour(k,eg,p)
    ctemp = production(k,kp,ltemp,eg,p) + numexpr.evaluate( '-kp + k * (1 - delt) + (1 + rs * exp(n)) * A - Ap' )
    cltemp =  composite_consumption(ctemp,ltemp,p)
    
    utemp = utility(cltemp,p)  
    discountemp = discount(cltemp,p)
    
    return np.array([utemp,discountemp,cltemp])

    

def new_value(ut,disc,cl,transit,V,pgrid):
    
    budget_not = (cl <= 0)
    
    EV0 = np.dot(transit,V).reshape(4, pgrid['nA'] , pgrid['nk'] )
    
    # EV0 transforms V0 in a matrix of shape (4,nA,nk,1)
    # It computes the expected future value of choosing one combination of assets
    # Given the present state (ie level of productivity and interest rate)
    
    TV0 = ut + disc * EV0[:,None,:,:,None]
    
    TV0[budget_not] = -9999999

    
    new_V0_temp = TV0.max(axis=3)
    new_V0      = new_V0_temp.max(axis=2)
    
    return   new_V0.reshape( (4 , pgrid['nA'] * pgrid['nk'] ) )
    


def find_value(k,kp,A,Ap,eg,n,p,pgrid,transit,V):
    
    # this function solves for the value function
    # it does not solve for the decision rule for A and k
    # this is done in the find_solution
    
    crit = 10
    V01 = V.copy()
    x = bud(k,kp,A,Ap,eg,n,p)
    

    while crit> p['epsilon']:
        
        TV = new_value(x[0],x[1],x[2],transit,V01,pgrid)
        crit = abs(V01 - TV).max()
        print crit
        V01 = TV
    
    return V01
        
#%prun -s time find_value(kgrid,kpgrid,Agrid,Apgrid,egrid,ngrid,p,pgrid,stoch_transit,V0)

Value = find_value(kgrid,kpgrid,Agrid,Apgrid,egrid,ngrid,p,pgrid,stoch_transit,V0)



def find_solution(k,kp,A,Ap,eg,n,p,pgrid,transit,V):
    
    # this function performs one more iteration on the value function
    # using this last iteration on the value function we are able to determine
    # the value of Aprime and kprime for given states A,k,e and n.
    
    
    # this step is identical to new_value
    
    delt = p['delta']
    omeg = p['omega']
    rs   = p['rstar']    
    ltemp = labour(k,eg,p)
    ctemp = production(k,kp,ltemp,eg,p)\
    + numexpr.evaluate( '-kp + k * (1 - delt) + (1 + rs * exp(n)) * A - Ap' )
    cltemp =  composite_consumption(ctemp,ltemp,p)
    budget_not = (cltemp <= 0)
    utemp = utility(cltemp,p)  
    disc = discount(cltemp,p)    
    EV = np.dot(transit,V).reshape(4, pgrid['nA'] , pgrid['nk'] )
    TV0 = utemp + disc * EV[:,None,:,:,None]    
    TV0[budget_not] = -9999999    
    new_V0_temp       = TV0.max(axis=3)
    new_V0_temp2      = new_V0_temp.max(axis=2) 
    
    
    # we then build a matrix made of the elements of V0
    # and we determine for which couple kprime and Aprime we found these values
    # this gives our decision rules
    
    new_V0_temp3      = np.repeat( new_V0_temp2 , pgrid['nk']*pgrid['nA'] , 1 ).reshape((4, pgrid['nA'],pgrid['nA'], pgrid['nk'] , pgrid['nk']))
    
    soltemp1          = ( new_V0_temp3 <> TV0 )    
    
    ksoltemp              = kp.copy()
    ksoltemp[soltemp1]     = 0
    
    Apsoltemp              = Ap.copy()
    Apsoltemp[soltemp1]     = -3
    
    ksoltemp2             = ksoltemp.max(axis=3)
    ksoltemp3             = ksoltemp2.max(axis=2)
    
    Apsoltemp2             = Apsoltemp.max(axis=3)
    Apsoltemp3             = Apsoltemp2.max(axis=2)
          
    return np.array([ksoltemp3,Apsoltemp3]) 

    
    
solution = find_solution(kgrid,kpgrid,Agrid,Apgrid,egrid,ngrid,p,pgrid,stoch_transit,Value)


kpdecided = solution[0]
Apdecided = solution[1]



# now we turn to the simulation of the model 

def simulation_output(kpd,Apd,pgrid,transit,nsim,kindex,Aindex,sindex):

    # we start with capital of index kindex
    # with foreign assets of index Aindex
    # with initial state of index sindex
    # we then simulate the model over nsim periods 

    from random import uniform as uni

    klin1 = np.linspace(pgrid['kmin'],pgrid['kmax'],pgrid['nk'])
    Alin1 = np.linspace(pgrid['Amin'],pgrid['Amax'],pgrid['nA'])

    ssim = np.zeros( nsim  )
    ksim = np.zeros( nsim + 1 ) 
    Asim = np.zeros( nsim + 1 )
    
    ssim[0] = sindex
    ksim[0] = klin1[kindex]
    Asim[0] = Alin1[Aindex]
    
    # first decision

    ksim[1] = kpdecided[sindex,Aindex,kindex]
    Asim[1] = Apdecided[sindex,Aindex,kindex]
    
    for i in xrange(1,nsim):
    
        draw = uni(0,1)
        kindex1 = np.where( klin1  == ksim[i] )[0][0]
        Aindex1 = np.where( Alin1  == Asim[i] )[0][0]
    
        if draw <= stoch_transit[ssim[i-1],0]:
            ssim[i] = 0
        elif  (draw > stoch_transit[ssim[i-1],0]) and (draw<= stoch_transit[ssim[i-1],0]+stoch_transit[ssim[i-1],1]):
            ssim[i] = 1
        elif  (draw > stoch_transit[ssim[i-1],0]+stoch_transit[ssim[i-1],1]) and (draw<= stoch_transit[ssim[i-1],0]+stoch_transit[ssim[i-1],1]+stoch_transit[ssim[i-1],2]):
            ssim[i] = 2
        else:
            ssim[i] = 3
    
        ksim[i+1] = kpdecided[ssim[i],Aindex1,kindex1]
        Asim[i+1] = Apdecided[ssim[i],Aindex1,kindex1]
    
    
    return np.array([ksim,Asim,ssim])
    

ns = 10000
    
simulation_result = simulation_output(kpdecided,Apdecided,pgrid,stoch_transit,ns,10,10,2)


#the result of our simulations can simply be expressed by:


k = simulation_result[0]
A = simulation_result[1]
s = simulation_result[2]
    
# we can plot this on a graph
    

from pylab import plot as plt

#for capital

t = np.linspace(0, ns ,ns + 1  )
#plt(t,k) 



#for investment

kk1 = k[0:ns]
kk2 = k[1:ns+1]
invsim = kk2 - (1 - p['delta']) * kk1


product = np.array([1.18/100,1.18/100, -1.18/100, -1.18/100] )
sp = s.astype(int)


l = labour(k[0:ns],product[sp],p)




"""
plt(t[0:ns],invsim)



"""

print np.std ( k/np.mean(k) )
print np.std ( invsim/np.mean(invsim) )
print np.std ( l/np.mean(l) )



