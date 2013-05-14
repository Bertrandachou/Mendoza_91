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

p = {'alpha': 0.32, 'rstar': 0.04, 'gamma': 1.001 , 'delta': 0.1, 'omega': 1.455, 'beta': 0.11, 'phi': 0}

# Then the grids for the stochastic process (transition and values)

p_stoch = {'e': 1.18, 'n': 0, 'rho': 0.36, 'rho_en': 0}

# We define the parameters of the grids

pgrid = {'kmin': 3.25 , 'kmax': 3.56 , 'nk': 10, 'Amin': -1.42 , 'Amax': 0.08 , 'nA': 12 }


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


klin =  np.array([np.linspace(pgrid['kmin'],pgrid['kmax'],pgrid['nk'])])
kgrid_temp = np.repeat( klin , 4 * pgrid['nA'] * pgrid['nA'] * pgrid['nk'], 0 )
kgrid      = np.reshape(kgrid_temp, (4 , pgrid['nA'], pgrid['nA'] , pgrid['nk'] , pgrid['nk']) )

# or (but apparently the first way is faster)
#kgrid = np.resize(klin,(4 , pgrid['nA'], pgrid['nA'] , pgrid['nk'] , pgrid['nk'] ))


#kplin      = np.repeat(klin, pgrid['nk'], 1)

kpgrid_temp = np.repeat(np.repeat(klin, pgrid['nk'], 1), 4 * pgrid['nA'] * pgrid['nA'], 0)
kpgrid      = np.reshape(kpgrid_temp, (4 , pgrid['nA'], pgrid['nA'] , pgrid['nk'] , pgrid['nk']) )


Alin  = np.linspace(pgrid['Amin'],pgrid['Amax'],pgrid['nA'])
Agrid_temp1 = np.array([np.repeat( Alin , (pgrid['nk']**2) * pgrid['nA'] , 0 )])
Agrid_temp2 = np.repeat( Agrid_temp1 , 4 , 0 )
Agrid       = np.reshape( Agrid_temp2 , (4 , pgrid['nA'], pgrid['nA'] , pgrid['nk'] , pgrid['nk']) )


Apgrid_temp1 = np.array([np.repeat( Alin , (pgrid['nk']**2)  , 0 )])
Apgrid_temp2 = np.array([np.repeat( Apgrid_temp1 , pgrid['nA'] , 0 )])
Apgrid_temp3 = np.repeat( Apgrid_temp2 , 4 , 0 )
Apgrid       = np.reshape( Apgrid_temp3 , (4 , pgrid['nA'], pgrid['nA'] , pgrid['nk'] , pgrid['nk']) )


egrid = np.zeros ( (4 , pgrid['nA'], pgrid['nA'] , pgrid['nk'] , pgrid['nk']) )
egrid[0:2].fill(p_stoch['e'])
egrid[2:].fill(-p_stoch['e'])



ngrid = np.zeros ( (4 , pgrid['nA'], pgrid['nA'] , pgrid['nk'] , pgrid['nk']) )
ngrid[0].fill(p_stoch['n'])
ngrid[1].fill(-p_stoch['n'])
ngrid[2].fill(p_stoch['n'])
ngrid[3].fill(-p_stoch['n'])



# now we define our functions 

# a function that computes the optimal labor as a function of k,e and the parameters p

def labour(k,e,p):
    return ( (1 - p['alpha']) * np.exp(e) * (k**p['alpha']) ) ** (1 / (p['alpha'] + p['omega'] - 1))
    

# utility as a function of c, l and the parameters p

def utility(c,l,p):
    if p['gamma'] == 1:
        return np.log( c - (l**p['omega'])/p['omega'] )
    else:
        return (( (c - (l**p['omega'])/p['omega'])**(1-p['gamma']) - 1 )) / (1 - p['gamma'])
    
# a function which returns the discount factor from c, l, e and the parameters p

def discount(c,l,p):
    return np.exp(-p['beta'] * np.log(1 + c - (l**p['omega'])/p['omega'] ))
    
    
# production function as a function k, k', l, e and the parameters p

def production(k,kp,l,e,p):
    return np.exp(e) * (k**p['alpha']) * (l**(1-p['alpha'])) - (p['phi']/2) * (kp - k)**2
    

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

def new_value(k,kp,A,Ap,e,n,p,pgrid,transit,V0):
    
    # we first need to compute the instantaneouse utility of the agent for every
    # state and possible decision he can take
    # when consumption is negative we put it to 0.001 and assigne a large negative utility
    # so that it is never optimal for the agent to choose these values
    
    ltemp = labour(k,e,p)
    ctemp = production(k,kp,ltemp,e,p) - kp + k * (1 - p['delta'])\
    + (1 + p['rstar'] * np.exp(n)) * A - Ap
    cltemp = ctemp - (ltemp**(p['omega']))/p['omega']
    
    
    budget_not = (cltemp <= 0)
    
    
    utemp = utility(ctemp,ltemp,p)
          
    disc = discount(ctemp,ltemp,p)
    
    
    
    EV0 = np.dot(transit,V0).reshape(4, pgrid['nA'] , pgrid['nk'] )
    
    # EV0 transforms V0 in a matrix of shape (4,nA,nk,1)
    # It computes the expected future value of choosing one combination of assets
    # Given the present state (ie level of productivity and interest rate)
    
    TV0 = utemp + disc * EV0[:,None,:,:,None]
    
    TV0[budget_not] = -99999999999

    
    new_V0_temp = TV0.max(axis=3)
    new_V0      = new_V0_temp.max(axis=2)
    
    return   new_V0.reshape( (4 , pgrid['nA'] * pgrid['nk'] ) )
    

#TV = new_value(kgrid,kpgrid,Agrid,Apgrid,egrid,ngrid,p,pgrid,stoch_transit,V0)


#test = new_value(kgrid,kpgrid,Agrid,Apgrid,egrid,ngrid,p,pgrid,stoch_transit,V0)

t1 = time.time()

for i in xrange(300):
    
    TV = new_value(kgrid,kpgrid,Agrid,Apgrid,egrid,ngrid,p,pgrid,stoch_transit,V0)
    
    #print abs(V0 - TV).max()
    
    V0 = TV

t2 = time.time()

print (t2-t1)    
#import pylab

#y = V0[0,0:pgrid['nk']].reshape((1,10))

#pylab.plot( klin , y )

