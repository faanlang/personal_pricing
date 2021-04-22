import numpy as np
import copy
from numpy.random import seed
from numpy.random import rand
import matplotlib.pyplot as plt

import operator 

#Github test update

util_a = 2
cost_a = 0.8
w = 1
cost_b = 0.5
util_b = cost_b + w * (util_a - cost_a) 
tech = 1.5
data_x =[]
data_xa = []
data_xb = []
data_a = []
data_b = []

data_a_u =[]
data_b_u =[]
data_xa_u =[]
data_xb_u =[]

data_a_r = []
data_b_r = []
data_xa_r =[]
data_xb_r =[]


xstar = 0.5 + ((util_a - cost_a)-(util_b-cost_b))/(2*tech)
x_U = 0.5 + ((util_a - cost_a)-(util_b-cost_b))/(6*tech)

x_a = (util_a - cost_a)/tech
x_b = (tech-(util_b - cost_b))/tech
if x_a > 1:
    x_a = 1
if x_b < 0:
    x_b = 0

def reserpriceA(x): 
    return util_a - tech*x
    
    
def reserpriceB(x):
    return util_b - tech *(1 -x)

def analytic_data():
    xline = np.linspace(0,1,201)
    for xn in xline:
        
        CSA_max = reserpriceA(xn) - cost_a
        CSB_max = reserpriceB(xn) - cost_b
        
        if CSA_max < 0 and CSB_max < 0:
            aprice_a = cost_a
            aprice_b = cost_b
        
        elif CSA_max > 0 or CSB_max > 0:
        
            if CSA_max > CSB_max:
                if CSB_max < 0:
                    aprice_a = reserpriceA(xn)
                    aprice_b = cost_b
                else:
                    aprice_a = reserpriceA(xn) - CSB_max
                    aprice_b = cost_b
        
            elif CSB_max > CSA_max:
                if CSA_max < 0:
                    aprice_b = reserpriceB(xn)
                    aprice_a = cost_a
                else:
                    aprice_b = reserpriceB(xn) - CSA_max
                    aprice_a = cost_a
                    
            else:
                aprice_a = cost_a
                aprice_b = cost_b
                
                
        else:
            aprice_a = cost_a
            aprice_b = cost_b
    
        data_x.append(xn)
        data_a.append(aprice_a)
        data_b.append(aprice_b)
    
def analytic_data_slopes()   :
    xstar = 0.5 + ((util_a - cost_a)-(util_b-cost_b))/(2*tech)
    xline_a = np.linspace(0,xstar,201)
    xline_b = np.linspace(xstar,1,201)
    for xn in xline_a:
        
        CSA_max = reserpriceA(xn) - cost_a
        CSB_max = reserpriceB(xn) - cost_b
        
        if CSA_max < 0 and CSB_max < 0:
            aprice_a = cost_a
            
        
        elif CSA_max > 0 or CSB_max > 0:
        
            if CSA_max > CSB_max:
                if CSB_max < 0:
                    aprice_a = reserpriceA(xn)
                    
                else:
                    aprice_a = reserpriceA(xn) - CSB_max
                    
        
            elif CSB_max > CSA_max:
                if CSA_max < 0:
                    
                    aprice_a = cost_a
                else:
                    
                    aprice_a = cost_a
                    
            else:
                aprice_a = cost_a
                
                
                
        else:
            aprice_a = cost_a
        data_xa.append(xn)
        data_a.append(aprice_a)
            
        
    for xn in xline_b:
        
        CSA_max = reserpriceA(xn) - cost_a
        CSB_max = reserpriceB(xn) - cost_b
        
        if CSA_max < 0 and CSB_max < 0:
            
            aprice_b = cost_b
        
        elif CSA_max > 0 or CSB_max > 0:
        
            if CSA_max > CSB_max:
                if CSB_max < 0:
                    
                    aprice_b = cost_b
                else:
                    
                    aprice_b = cost_b
        
            elif CSB_max > CSA_max:
                if CSA_max < 0:
                    aprice_b = reserpriceB(xn)
                    
                else:
                    aprice_b = reserpriceB(xn) - CSA_max
                    
                    
            else:
                
                aprice_b = cost_b
                
                
        else:
            
            aprice_b = cost_b
            
        
    
        data_xb.append(xn)
        data_b.append(aprice_b)
        


def uniform_prices():
    x_U = 0.5 + ((util_a - cost_a)-(util_b-cost_b))/(6*tech)
    uline_a = np.linspace(0,x_U,201)
    uline_b = np.linspace(x_U,1,201)
    for xn in uline_a:
        uprice_a = cost_a + tech + ((util_a - cost_a)-(util_b-cost_b))/(3)
        data_xa_u.append(xn)
        data_a_u.append(uprice_a)
    for xn in uline_b:
        uprice_b = cost_b + tech + ((util_a - cost_a)-(util_b-cost_b))/(3)
        data_xb_u.append(xn)
        data_b_u.append(uprice_b)
    
    
def reserprices():
    x_a = (util_a - cost_a)/tech
    x_b = (tech-(util_b - cost_b))/tech

    if x_a > 1:
        x_a = 1
    if x_b < 0:
        x_b = 0
    rline_a = np.linspace(0,x_a,201)
    rline_b = np.linspace(x_b,1,201)
    for xn in rline_a:
        data_a_r.append(reserpriceA(xn))
        data_xa_r.append(xn)
    for xn in rline_b:
        data_b_r.append(reserpriceB(xn))
        data_xb_r.append(xn)
     
analytic_data()
#analytic_data_slopes()
uniform_prices()
reserprices()

plt.plot(data_xa_r,data_a_r, color='cornflowerblue',  label = r'$r^A(x)$')
plt.plot(data_xb_r,data_b_r,color='wheat',label = r'$r^B(x)$')

#plt.plot(data_x,data_a, label = r'$p^A(x)$')
#plt.plot(data_x,data_b,label = r'$p^B(x)$')

#plt.plot(data_xa_u,data_a_u, label = r'$p^A_U$')
#plt.plot(data_xb_u,data_b_u,label = r'$p^B_U$')

plt.axvline(x=x_a,color='seashell', linestyle='dashed')
plt.axvline(x=x_b,color='seashell', linestyle='dashed')



plt.axhline(y=cost_a, color='magenta', linestyle='dashed')
plt.axhline(y=cost_b, color='darkorange', linestyle='dashed')
#plt.axvline(x=xstar,color='seashell', linestyle='dashed')
#plt.axvline(x=x_U,color='seashell', linestyle='dashed')
#plt.xticks([0,xstar,x_U,1], [0,r'$x^*$',r'$x^*_U$',1])

#plt.xticks([0,x_b,xstar,x_a,1], [0,r'$\bar{x}^B$', r'$x^*$',r'$\bar{x}^A$',1])
#plt.xticks([0,x_U,1], [0,r'$x^*_U$',1])

plt.xticks([0,x_b,x_a,1], [0,r'$\bar{x}^B$',r'$\bar{x}^A$',1])


plt.yticks([cost_a,cost_b], [r'$c^A$',r'$c^B$'])
plt.xlabel(r'$x$')
plt.ylabel(r'$p$')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05),ncol=3, fancybox=True, shadow=True)

#plt.legend(bbox_to_anchor=(1.1, 1.05))

plt.ylim(0,2.5)
plt.xlim(0,1)
plt.savefig('HotellingRES.png')



     # for xn in xline_b:
        
     #    CSA_max = reserpriceA(xn) - cost_a
     #    CSB_max = reserpriceB(xn) - cost_b
        
     #    if CSA_max < 0 and CSB_max < 0:
     #        aprice_a = cost_a
     #        aprice_b = cost_b
        
     #    elif CSA_max > 0 or CSB_max > 0:
        
     #        if CSA_max > CSB_max:
     #            if CSB_max < 0:
     #                aprice_a = reserpriceA(xn)
     #                aprice_b = cost_b
     #            else:
     #                aprice_a = reserpriceA(xn) - CSB_max
     #                aprice_b = cost_b
        
     #        elif CSB_max > CSA_max:
     #            if CSA_max < 0:
     #                aprice_b = reserpriceB(xn)
     #                aprice_a = cost_a
     #            else:
     #                aprice_b = reserpriceB(xn) - CSA_max
     #                aprice_a = cost_a
                    
     #        else:
     #            aprice_a = cost_a
     #            aprice_b = cost_b
                
                
     #    else:
     #        aprice_a = cost_a
     #        aprice_b = cost_b
            