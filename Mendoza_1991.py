from __future__ import division

# -*- coding: utf-8 -*-
"""
Created on Tue May 07 17:58:10 2013

@author: Bertrand Achou
"""

""" This is an attempt to reproduce the results obtained by Mendoza
in his paper 'RBC in a small open economy' published in the AER in 1991.
We use value iteration with vectorization."""

import numpy as np

# First, we define our grid of parameters

p = {'alpha': 0.32, 'rstar': 0.04, 'gamma': 2 , 'delta': 0.1, 'omega': 1.455, 'beta': 0.11, 'phi': 0.028, 'epsilon': 10**(-6) }

# Then the grids for the stochastic process (transition and values)

p_stoch = {'ep': 1.29/100, 'n': 0, 'rho': 0.42, 'rho_en': 0}

# We define the parameters of the grids

pgrid = {'kmin': 3.25, 'kmax': 3.56 , 'nk': 22, 'Amin': -1.42 , 'Amax': 0.08 , 'nA': 22 }

# We build the transition matrix that will be used
# NB : There is a mistake in the paper 803 about these matrices
# I explain it in more details in the document for this project
# However this mistake is consitent with what Mendoza says he does 
# in the rest of the paper

Pi          = (p_stoch['rho_en'] + 1) / 4

stoch_transit = np.array([\
[(1-p_stoch['rho']) * Pi + p_stoch['rho'], (1-p_stoch['rho']) * (0.5 - Pi), (1-p_stoch['rho']) * (0.5 - Pi), (1-p_stoch['rho']) * Pi],\
[(1-p_stoch['rho']) * Pi, (1-p_stoch['rho']) * (0.5 - Pi) + p_stoch['rho'] , (1-p_stoch['rho']) * (0.5 - Pi), (1-p_stoch['rho']) * Pi],\
[(1-p_stoch['rho']) * Pi, (1-p_stoch['rho']) * (0.5 - Pi), (1-p_stoch['rho']) * (0.5 - Pi) + p_stoch['rho'], (1-p_stoch['rho']) * Pi],\
[(1-p_stoch['rho']) * Pi, (1-p_stoch['rho']) * (0.5 - Pi), (1-p_stoch['rho']) * (0.5 - Pi), (1-p_stoch['rho']) * Pi + p_stoch['rho']]\
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


# now we define some of the functions that will be used in the rest of the program 

import numexpr

# a function that computes the optimal labor as a function of k,e and the parameters p

def labour(k,eg,p):
    # computes the optimal labor as a function of k,e 
    # and the parameters p
    alpha = p['alpha']
    omega = p['omega']
    return numexpr.evaluate( '( (1 - alpha) * exp(eg) * (k**alpha) ) ** (1 / ( alpha + omega - 1))' )
    
def composite_consumption(c,l,p):
    # computes what I call composite consumption 
    # which is a composite of regular consuption minus the 
    # disutility of labour
    omega = p['omega']
    return numexpr.evaluate( 'c - (l**omega)/omega' )
    
def utility(cc,p):
    # utility as a function of composite consumption 
    # and the parameters p
    gamma = p['gamma']
    if gamma == 1:
        return numexpr.evaluate( 'log(cc)' )
    else:
        return numexpr.evaluate( '( (cc)**(1-gamma) - 1 ) / (1 - gamma)' )
        
def discount(cc,p):
    #returns the discount factor from composite consumption 
    # and the parameters p
    beta = p['beta']
    return numexpr.evaluate( 'exp(-beta * log(1 + cc ))' )
    
    
# production function as a function k, k', l, e and the parameters p

def production(k,kp,l,eg,p):
    # computes the level of production (taking into account the cost of 
    # adjustment of capital) as a function of k, k', l, e and the parameters
    # p 
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
# it depends on k, k', A, A', e, n ,
# the grid of parameters p,
# the transition matrix of the stochastic process
# and on the initial guess for the value function
# this is done in three steps to avoid useless calculations
# for instance bud computes values which need to be re-used at 
# each iterations and so do not need to be computed each time  

def bud(k,kp,A,Ap,eg,n,p):
    
    # this computes the instantaneouse utility of the agent
    # her discount rate
    # and her level of composite consumption
    # for every state and possible decisions he can take
    
    delt = p['delta']
    omeg = p['omega']
    rs   = p['rstar']
    
    ltemp = labour(k,eg,p)
    ctemp = production(k,kp,ltemp,eg,p)\
    + numexpr.evaluate( '-kp + k * (1 - delt) + (1 + rs * exp(n)) * A - Ap' )
    cltemp =  composite_consumption(ctemp,ltemp,p)
    
    utemp = utility(cltemp,p)  
    discountemp = discount(cltemp,p)
    
    return np.array([utemp,discountemp,cltemp])

    

def new_value(ut,disc,cl,transit,V,pgrid):
    
    # computes the new value function for a initial guess V
    # it depends on the values found using the bud function 
    # and on the transition probability matrix
    # when composite consumption is negative
    # the utility associated with this decision is set to a very
    # large negative number so it is never optimal to choose 
    # decision rules associated with negative consumption
    
    budget_not = (cl <= 0)
    
    # EV0 transforms V0 in a matrix of shape (4,nA,nk,1)
    # It computes the expected future value of choosing one combination of assets
    # Given the present state (ie level of productivity and interest rate)
    # It is the result of a simple matrix product
    
    EV0 = np.dot(transit,V).reshape(4, pgrid['nA'] , pgrid['nk'] )   
    
    # We then need to compute the expected value associated with every decision
    # and initial state. For this we need to extend the EV0 matrix along
    # the (non-existing) dimension for A and k
    
    TV0 = ut + disc * EV0[:,None,:,:,None]    
    TV0[budget_not] = -9999999
    
    # We then take the maximum of these values along the axis for 
    # A' and k' (the 3rd and 4th axis)
    
    new_V0      = TV0.max(axis=3).max(axis=2)
    
    return   new_V0.reshape( (4 , pgrid['nA'] * pgrid['nk'] ) )
    

def find_value(k,kp,A,Ap,eg,n,p,pgrid,transit,V):
    
    # this function solves for the value function
    # using the previously defined functions
    # bud and new_value
    # it returns the value function for a given precision
    # epsilon set in the grid of parameters
    
    crit = 10
    V01 = V.copy()
    x = bud(k,kp,A,Ap,eg,n,p)
    
    while crit> p['epsilon']:
        TV = new_value(x[0],x[1],x[2],transit,V01,pgrid)
        crit = abs(V01 - TV).max()
        #print crit
        V01 = TV
    
    return V01
    
    
# we then can compute our value function Value
        
#%prun -s time find_value(kgrid,kpgrid,Agrid,Apgrid,egrid,ngrid,p,pgrid,stoch_transit,V0)

Value = find_value(kgrid,kpgrid,Agrid,Apgrid,egrid,ngrid,p,pgrid,stoch_transit,V0)



def find_solution(k,kp,A,Ap,eg,n,p,pgrid,transit,V):
    
    # this function performs one more iteration on the value function
    # using this last iteration on the value function we are able to determine
    # the value of Aprime and kprime for given states A,k,e and n.
    
    
    # this first step is identical to new_value
    
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
    new_V0_temp       = TV0.max(axis=3).max(axis=2) 
    
    
    # we then build a matrix made of the elements of V0
    # and we determine for which couple kprime and Aprime we found these values
    # this gives our decision rules
    
    new_V0_temp2      = np.repeat( new_V0_temp , pgrid['nk']*pgrid['nA'] , 1 ).reshape((4, pgrid['nA'],pgrid['nA'], pgrid['nk'] , pgrid['nk']))
    soltemp1          = ( new_V0_temp2 <> TV0 )    
    ksoltemp              = kp.copy()
    ksoltemp[soltemp1]     = 0
    Apsoltemp              = Ap.copy()
    Apsoltemp[soltemp1]     = -3

    ksoltemp2              = ksoltemp.max(axis=3).max(axis=2)    
    Apsoltemp2             = Apsoltemp.max(axis=3).max(axis=2)
          
    return np.array([ksoltemp2,Apsoltemp2]) 


# From this we get our solutions for k' and A'

solution = find_solution(kgrid,kpgrid,Agrid,Apgrid,egrid,ngrid,p,pgrid,stoch_transit,Value)

kpdecided = solution[0]
Apdecided = solution[1]


# We can then turn to the simulation of the model 

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
    
# we then determinate the number of simulations
    
ns = 100000    
simulation_result = simulation_output(kpdecided,Apdecided,pgrid,stoch_transit,ns,10,10,2)


# the result of our simulations can simply be expressed by:

k = simulation_result[0]
A = simulation_result[1]
s = simulation_result[2]


# From these results we can compute the different variable of interest

sp = s.astype(int)

stochastic_e = np.array([p_stoch['ep'],p_stoch ['ep'],-p_stoch ['ep'],-p_stoch ['ep'] ])
stochastic_n = np.array([p_stoch['n'],-p_stoch ['n'],p_stoch ['n'],-p_stoch ['n'] ])
s_e = stochastic_e[sp]
s_n = stochastic_n[sp]

# Investment
kk1 = k[0:ns]
kk2 = k[1:ns+1]
inv = kk2 - (1 - p['delta']) * kk1

# Hours worked
sp = s.astype(int)
labour_s = labour(k[0:ns],stochastic_e[sp],p)


# Output
gdp = production(kk1,kk2,labour_s,s_e,p)

# GNP
gnp = gdp + p['rstar']*s_n*A[0:ns] 

# Consumption
consumption = gdp - kk2 + (1 - p['delta'])*kk1 + (1+p['rstar']*s_n)*A[0:ns] - A[1:(ns+1)]

# savings 
savings = gdp - consumption

# TB / Y 
TB_Y = (A[1:(ns+1)] -(1+p['rstar']*s_n)*A[0:ns]) / gdp 


# Here are the results for the standard deviations
# All these results are ordered as follow
# [gdp,gnp,consumption,savings,investement,capital,labour,TB/Y]


sd_gdp           = np.std ( gdp/np.mean(gdp) ) *100
sd_gnp           = np.std ( gnp/np.mean(gnp) ) *100
sd_consumption   = np.std ( consumption/np.mean(consumption) )*100
sd_savings       = np.std ( savings/np.mean(savings) ) *100
sd_investment    = np.std ( inv/np.mean(inv) ) *100
sd_capital       = np.std ( k/np.mean(k) ) *100
sd_labour        = np.std ( labour_s/np.mean(labour_s) ) *100
sd_TB_Y          = np.std ( TB_Y ) *100

sd = np.array([ sd_gdp, sd_gnp , sd_consumption, sd_savings,\
sd_investment, sd_capital, sd_labour, sd_TB_Y ]) 


# For the first-order autcorrelations

ac_gdp           = np.corrcoef( np.array([gdp[0:ns-1],gdp[1:ns]]) )[0,1] 
ac_gnp           = np.corrcoef( np.array([gnp[0:ns-1],gnp[1:ns]]) )[0,1] 
ac_consumption   = np.corrcoef( np.array([consumption[0:ns-1],consumption[1:ns]]) )[0,1] 
ac_savings       = np.corrcoef( np.array([savings[0:ns-1],savings[1:ns]]) )[0,1]  
ac_investment    = np.corrcoef( np.array([inv[0:ns-1],inv[1:ns]]) )[0,1]  
ac_capital       = np.corrcoef( np.array([k[0:ns-1],k[1:ns]]) )[0,1]  
ac_labour        = np.corrcoef( np.array([labour_s[0:ns-1],labour_s[1:ns]]) )[0,1] 
ac_TB_Y          = np.corrcoef( np.array([TB_Y[0:ns-1],TB_Y[1:ns]]) )[0,1] 

ac = np.array([ ac_gdp, ac_gnp , ac_consumption, ac_savings,\
ac_investment, ac_capital, ac_labour, ac_TB_Y ]) 


# correlation with gdp

corr_gdp         = np.corrcoef( np.array([gdp,gdp]) )[0,1] 
corr_gnp         = np.corrcoef( np.array([gnp,gdp]) )[0,1] 
corr_consumption = np.corrcoef( np.array([consumption,gdp]) )[0,1] 
corr_savings     = np.corrcoef( np.array([savings,gdp]) )[0,1]  
corr_investment  = np.corrcoef( np.array([inv,gdp]) )[0,1]  
corr_capital     = np.corrcoef( np.array([k[0:ns],gdp]) )[0,1]  
corr_labour      = np.corrcoef( np.array([labour_s,gdp]) )[0,1]  
corr_TB_Y        = np.corrcoef( np.array([TB_Y,gdp]) )[0,1]  

corr_w_gdp = np.array([ corr_gdp, corr_gnp , corr_consumption, corr_savings,\
corr_investment, corr_capital, corr_labour, corr_TB_Y ])

"""
# We can check the value for the different coefficient of correlations for the stochastic process
# and see that our matrix of transition is in line with what is done in the paper

# correlation of e and n 

correlation_e_n = np.corrcoef( np.array([s_e,s_n]) )

# auto-correlation of e and n

correlation_e_e = np.corrcoef( np.array([s_e[0:ns-1],s_e[1:ns]]) )
correlation_n_n = np.corrcoef( np.array([s_n[0:ns-1],s_n[1:ns]]) )

"""