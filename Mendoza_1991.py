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

p = {'alpha': 0.32, 'rstar': 0.04, 'gamma': 4 , 'delta': 0.1, 'omega': 1.455, 'beta': 0.11, 'phi': 0, 'epsilon': 10**(-2) }

# Then the grids for the stochastic process (transition and values)

p_stoch = {'e': 1.18/100, 'n': 0, 'rho': 0.36, 'rho_en': 0}

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
egrid[2:].fill(- p_stoch['e'])



ngrid = np.zeros ( (4 , pgrid['nA'], pgrid['nA'] , pgrid['nk'] , pgrid['nk']) )
ngrid[0].fill(p_stoch['n'])
ngrid[1].fill(-p_stoch['n'])
ngrid[2].fill(p_stoch['n'])
ngrid[3].fill(-p_stoch['n'])



# now we define our functions 

import numexpr

# a function that computes the optimal labor as a function of k,e and the parameters p

def labour(k,e,p):
    alpha = p['alpha']
    omega = p['omega']
    return numexpr.evaluate( '( (1 - alpha) * exp(e) * (k**alpha) ) ** (1 / (alpha + omega- 1))' )
    

# utility as a function of c, l and the parameters p

def utility(c,l,p):
    gamma = p['gamma']
    omega = p['omega']
    if gamma == 1:
        return numexpr.evaluate( 'log( c - (l**omega)/omega )' )
    else:
        return numexpr.evaluate( '( (c - (l**omega)/omega)**(1-gamma) - 1 ) / (1 - gamma)' )
    
# a function which returns the discount factor from c, l, e and the parameters p

def discount(c,l,p):
    beta = p['beta']
    omega = p['omega']
    return numexpr.evaluate( 'exp(-beta * log(1 + c - (l**omega)/omega ))' )
    
    
# production function as a function k, k', l, e and the parameters p

def production(k,kp,l,e,p):
    alpha = p['alpha']
    phi   = p['phi']
    return  numexpr.evaluate( 'exp(e) * (k**alpha) * (l**(1-alpha)) - (phi/2) * (kp - k)**2' )
    

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

def new_value(k,kp,A,Ap,e,n,p,pgrid,transit,V):
    
    # we first need to compute the instantaneouse utility of the agent for every
    # state and possible decision he can take
    # when consumption is negative we put it to 0.001 and assigne a large negative utility
    # so that it is never optimal for the agent to choose these values
    delt = p['delta']
    omeg = p['omega']
    rs   = p['rstar']
    
    ltemp = labour(k,e,p)
    ctemp = production(k,kp,ltemp,e,p) \
    + numexpr.evaluate( '-kp + k * (1 - delt) + (1 + rs * exp(n)) * A - Ap' )
    
    cltemp =  numexpr.evaluate( 'ctemp - (ltemp**(omeg))/omeg' )
    
    
    budget_not = (cltemp <= 0)
    
    
    utemp = utility(ctemp,ltemp,p)
          
    disc = discount(ctemp,ltemp,p)
    
    
    
    EV0 = np.dot(transit,V).reshape(4, pgrid['nA'] , pgrid['nk'] )
    
    # EV0 transforms V0 in a matrix of shape (4,nA,nk,1)
    # It computes the expected future value of choosing one combination of assets
    # Given the present state (ie level of productivity and interest rate)
    
    TV0 = utemp + disc * EV0[:,None,:,:,None]
    
    TV0[budget_not] = -9999999

    
    new_V0_temp = TV0.max(axis=3)
    new_V0      = new_V0_temp.max(axis=2)
    
    return   new_V0.reshape( (4 , pgrid['nA'] * pgrid['nk'] ) )
    

test = new_value(kgrid,kpgrid,Agrid,Apgrid,egrid,ngrid,p,pgrid,stoch_transit,V0)


def find_value(kgrid,kpgrid,Agrid,Apgrid,egrid,ngrid,p,pgrid,stoch_transit,V0):
    
    crit = 10
    iteration = 0

    while crit> p['epsilon']:
        
        #iteration = iteration + 1       
    
        TV = new_value(kgrid,kpgrid,Agrid,Apgrid,egrid,ngrid,p,pgrid,stoch_transit,V0)
    
        crit = abs(V0 - TV).max()
        # print iteration
        print crit
    
        V0 = TV
    
    return V0
        
#%prun -s time find_value(kgrid,kpgrid,Agrid,Apgrid,egrid,ngrid,p,pgrid,stoch_transit,V0)

Value = find_value(kgrid,kpgrid,Agrid,Apgrid,egrid,ngrid,p,pgrid,stoch_transit,V0)



def find_solution(k,kp,A,Ap,e,n,p,pgrid,transit,V):
    
    delt = p['delta']
    omeg = p['omega']
    rs   = p['rstar']
    
    ltemp = labour(k,e,p)
    ctemp = production(k,kp,ltemp,e,p) \
    + numexpr.evaluate( '-kp + k * (1 - delt) + (1 + rs * exp(n)) * A - Ap' )
    
    cltemp =  numexpr.evaluate( 'ctemp - (ltemp**(omeg))/omeg' )
    
    
    budget_not = (cltemp <= 0)
    
    
    utemp = utility(ctemp,ltemp,p)
          
    disc = discount(ctemp,ltemp,p)
    
    
    
    EV = np.dot(transit,V).reshape(4, pgrid['nA'] , pgrid['nk'] )
    
    # EV0 transforms V0 in a matrix of shape (4,nA,nk,1)
    # It computes the expected future value of choosing one combination of assets
    # Given the present state (ie level of productivity and interest rate)
    
    TV0 = utemp + disc * EV[:,None,:,:,None]
    
    TV0[budget_not] = -9999999

    
    new_V0_temp       = TV0.max(axis=3)
    
    print np.shape(new_V0_temp)
    
    new_V0_temp2      = new_V0_temp.max(axis=2) 
    
    new_V0_temp3      = new_V0_temp2.reshape( (4, pgrid['nA'], pgrid['nk']) ) 
    new_V0_temp4      = np.repeat( new_V0_temp3 , pgrid['nk']*pgrid['nA'] , 1 ).reshape((4, pgrid['nA'],pgrid['nA'], pgrid['nk'] , pgrid['nk']))
    
    #print TV0 
    
    soltemp1          = ( new_V0_temp4 <> TV0 )
    #soltemp2          = ( new_V0_temp4 == TV0 )
    
    
    ksoltemp              = kp.copy()
    ksoltemp[soltemp1]     = 0
    #ksoltemp[soltemp2]     = 1
    
    #Apsoltemp              = Ap.copy()
    #Apsoltemp[soltemp1]     = -3
    #ksoltemp[soltemp2]     = 1    

    
    ksoltemp2             = ksoltemp.max(axis=3)
    ksoltemp3             = ksoltemp2.max(axis=2)
    
    #Apsoltemp2             = Apsoltemp.max(axis=3)
    #Apsoltemp3             = Apsoltemp2.max(axis=2)
    
   
    
          
    return ksoltemp3 #Apsoltemp3 

    
    
solution = find_solution(kgrid,kpgrid,Agrid,Apgrid,egrid,ngrid,p,pgrid,stoch_transit,Value)


x = solution.reshape( ( 4 , pgrid['nk']*pgrid['nA']) )



import pylab



Vplot = x[0,0*pgrid['nk']:(0*pgrid['nk']+pgrid['nk'])].reshape(pgrid['nk'])
"""
Vplot1 = x[3,1*pgrid['nk']:(1*pgrid['nk']+pgrid['nk'])].reshape(pgrid['nk'])


Vplot2 = x[3,2*pgrid['nk']:(2*pgrid['nk']+pgrid['nk'])].reshape(pgrid['nk'])
Vplot3 = x[3,3*pgrid['nk']:(3*pgrid['nk']+pgrid['nk'])].reshape(pgrid['nk'])
Vplot4 = x[3,4*pgrid['nk']:(4*pgrid['nk']+pgrid['nk'])].reshape(pgrid['nk'])
Vplot5 = x[3,5*pgrid['nk']:(5*pgrid['nk']+pgrid['nk'])].reshape(pgrid['nk'])
Vplot6 = x[3,6*pgrid['nk']:(6*pgrid['nk']+pgrid['nk'])].reshape(pgrid['nk'])
Vplot7 = x[3,7*pgrid['nk']:(7*pgrid['nk']+pgrid['nk'])].reshape(pgrid['nk'])


kplot = klin.reshape(pgrid['nk'])


pylab.plot(kplot,Vplot)
"""pylab.plot(kplot,Vplot1)
pylab.plot(kplot,Vplot2)
pylab.plot(kplot,Vplot3)
pylab.plot(kplot,Vplot4)
pylab.plot(kplot,Vplot5)
pylab.plot(kplot,Vplot6)
pylab.plot(kplot,Vplot7)


"""
    

"""

import pylab

Vplot = Value[3,4*pgrid['nk']:(4*pgrid['nk']+pgrid['nk'])].reshape(pgrid['nk'])
kplot = klin.reshape(pgrid['nk'])

pylab.plot(kplot,Vplot)

"""