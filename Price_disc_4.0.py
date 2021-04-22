import numpy as np
import copy
from numpy.random import seed
from numpy.random import rand
import matplotlib.pyplot as plt
import matplotlib
import configparser

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression

import operator 
import random

# THE FUNCTION simulationrunner ran all the way at end does the actual simulations

#random seed, put some arbitrary number to change this
seed(4444)
random.seed(4444)


solve_int = 0.001
firmA = []
firmB = []
CS = []
results = []
max_price = 3


#most important function, reads a configfile and runs a simulation. Option to plot
def simulation(configname,plots):  
    config = configparser.ConfigParser()
    config.sections()
    global util_a, cost_a,w,cost_b,tech,precision,T,training, util_b
    global history_a, history_b, history_g, data_x, data_a, data_b, profit_a, profit_b
    config.read(configname)
    
    util_a = float(config['parameters']['util_a'])
    cost_a = float(config['parameters']['cost_a'])
    w = float(config['parameters']['w'])
    cost_b = float(config['parameters']['cost_b'])
    tech = float(config['parameters']['tech'])
    precision = float(config['parameters']['precision'])
    
    T = int(config['parameters']['T'])
    training = int(config['parameters']['training']) 
    runtime  = T + training
    
    util_b = cost_b + w * (util_a - cost_a)      
    t = 0
    
    history_a = [[]]
    history_b = [[]]
    history_g = [[]]
    profit_a = []
    profit_b = []
    cons_surplus_a = []
    cons_surplus_b = []
    

    del history_a[0]
    del history_b[0]
    
    data_x = []
    data_a = []
    data_b = []
    
    global uniformprice_b 
    uniformprice_b = cost_b + 0.5 *(tech - ((util_a-cost_a)-(util_b-cost_b)))
    
    #the central While loop
    while t < runtime:
        xx = rand(1)[0]
        
        if t< training:
            finalprice_a = random.uniform(cost_a, max_price)
            finalprice_b = random.uniform(cost_b, max_price)
            #finalprice_b = cost_b
        
        else:
     
            finalprice_a = priceSOLVER(xx, history_a, cost_a)
            finalprice_b = priceSOLVER(xx, history_b, cost_b)
            #finalprice_b = cost_b
            
        
        consumerchoice(xx, finalprice_a, finalprice_b)
        t +=1
    
    #print(history_a)
    analytic_data()
        
    succes_a_x_plot=[]
    succes_a_p_plot=[]
    fail_a_x_plot=[]
    fail_a_p_plot=[]
    succes_b_x_plot=[]
    succes_b_p_plot=[]
    fail_b_x_plot=[]
    fail_b_p_plot=[]
    
    #remove training from history, we dont plot that
    del history_a[:(training)]
    del history_b[:(training)]
    
    for items in history_a:
        if items[3] == 1:
            succes_a_x_plot.append(items[0])
            succes_a_p_plot.append(items[1])
            #print('A SUCCES!')
            profit_a.append(items[1]-cost_a)
            cons_surplus_a.append(items[4])
        if items[3] == 0:
            fail_a_x_plot.append(items[0])
            fail_a_p_plot.append(items[1])
            
    for items in history_b:
        if items[3] == 1:
            succes_b_x_plot.append(items[0])
            succes_b_p_plot.append(items[1])
            profit_b.append(items[1]-cost_b)
            cons_surplus_b.append(items[4])
        if items[3] == 0:
            fail_b_x_plot.append(items[0])
            fail_b_p_plot.append(items[1])
            
    
    
    #Plotting
    if plots == True:
        plt.scatter(succes_a_x_plot, succes_a_p_plot, s=2,  label= 'A +')
        plt.scatter(fail_a_x_plot, fail_a_p_plot,s=5,linewidth=0.2,marker = "o", facecolor = 'none',edgecolor = 'darkorange',  label = 'A -')
        plt.scatter(succes_b_x_plot, succes_b_p_plot,s=2, label = 'B +', color = "limegreen")
        plt.scatter(fail_b_x_plot, fail_b_p_plot,s=5,linewidth=0.2,marker = "o",facecolor = 'none',edgecolor = 'red', label = 'B -' )
        plt.plot(data_x,data_a, color = 'magenta')
        plt.plot(data_x,data_b, color = 'black')
        plt.axhline(y=cost_a, color='magenta', linestyle='dashed')
        plt.axhline(y=cost_b, color='black', linestyle='dashed')
        
        #plt.plot(data_x, reserpriceA(data_x), color='tan', linestyle='dashed')
        
        
        plt.text(0.01, cost_a- 0.2, r'$c^A$')
        plt.text(0.95, cost_b-0.2, r'$c^B$')
        
        plt.ylim(0,3)
        plt.xlim(0,1)
        plt.xlabel(r'$x$')
        plt.ylabel(r'$p$')
        
        plt.legend()
        plotname = 'Ua' +str(util_a) + 'Ca'+str(cost_a)+'w'+ str(w)+'Cb'+str(cost_b)+'t'+str(tech)+'T'+str(T)+'tr'+str(training)+'lmd'+str(precision)
        plt.title(r'$\omega = $'+str(w)+'  ,  '+r'$c^B = $'+ str(cost_b))
        
        #name = 'A_Con_Mon_t' + str(t) + '_uB' + str(util_b)
        name = 'Plot' + plotname
        plt.savefig(name+'1'+ '.pdf')
        plt.show()
    #avg_price_a = (sum(succes_a_p_plot) + sum(fail_a_p_plot) )/ (len(succes_a_p_plot)+len(fail_a_p_plot))
    #avg_price_b = (sum(succes_b_p_plot) + sum(fail_b_p_plot) )/ (len(succes_b_p_plot)+len(fail_b_p_plot))
    finalprofit_A = sum(profit_a)
    finalprofit_B = sum(profit_b)
    ppc_A = sum(profit_a)/T
    ppc_B = sum(profit_b)/T
    finalconsurp = sum(cons_surplus_a)+sum(cons_surplus_b)
    ppc_CS  = (sum(cons_surplus_a)+sum(cons_surplus_b))/T
    
    market_share_a = len(succes_a_x_plot)/T
    market_share_b = len(succes_b_x_plot)/T
    market_share_o = 1 - market_share_a - market_share_b

    
    #returning some info for the bar plots
    return[ppc_A,ppc_B,ppc_CS,market_share_a,market_share_b,market_share_o]

###############################################################################
    
#functions that calculate the reservation price for a given x for each firm's product
def reserpriceA(x): 
    return util_a - tech*x
    
    
def reserpriceB(x):
    return util_b - tech *(1 -x)

##############################################################################

#this function simulates the consumer choice
def consumerchoice(x,price_a,price_b):
    rng = rand(1)[0]
    csa = reserpriceA(x) - price_a
    csb = reserpriceB(x) - price_b
    #print('csa, csab = ', csa,csb)
    #print('x =  ', x)
    k_a = 0
    k_b = 0
    k_g = '0'
    
    #print('teller  '  , np.exp(precision*csa))
    #print('noemer ',1 + np.exp(precision*csa)+np.exp(precision*csb) )
    
    
    # calculate the probabilities that consumer x will by at either firm A or B
    PA = np.exp(precision*csa) / (1 + np.exp(precision*csa)+np.exp(precision*csb))
   
    PB = np.exp(precision*csb) / (1 + np.exp(precision*csa)+np.exp(precision*csb))
    
    # k_g is coded with (3,4,5) for (A,B,N) respectively (we dont really use this)
    # k_a and k_b are coded (1 = buy at that firm, 0 = buy not at that firm)
    #print('rng , PA, PB: ', rng, PA, PB)
    if csa > 0 and csb > 0:
        if rng < PA:
            k_a = 1
            k_g = 'A'
            #print(price_a - cost_a)
            
        elif rng < PA + PB:
            k_b = 1
            k_g = 'B'
            
        else: 
            k_g = 'N'
        #print(rng)
       
        
    elif csa > 0:
        if rng < PA:
            k_a = 1
            k_g = 'A'
            #print(price_a - cost_a)
        
    elif csb > 0:
        if rng < PB:
            k_b = 1
            k_g = 'B'
    
    #add result to history
    history_a.append([x,price_a,price_b,k_a,csa])
    history_b.append([x,price_b,price_a,k_b,csb])
    history_g.append([x,price_a,price_b,k_g])
    
    return  [PA, PB]

###############################################################################



#gives an OLS estimation of the price of opposing firm (best response function)
def priceOLS(history):
    opposing_price = []
    h_ols = copy.deepcopy(history)
    del h_ols[:(training-1)]
    #remove the explained variable from the array
    #remove the k_i from the array, as we do not need it for the OLS
    for items in h_ols:
        del items[4]
        opposing_price.append(items[2])
        del items[2]
        del items[2]
    ols_reg_price = LinearRegression().fit(h_ols, opposing_price)
    return [ols_reg_price.intercept_,ols_reg_price.coef_[0],ols_reg_price.coef_[1]]

#gives a Logit estimation of the probability at which 
#the consumer x is going to buy at your firm given prices of both firms
def probLOGIT(history):
    observed_choice = []
    h_logit = copy.deepcopy(history)
    #remove the explained variable from the array
    #remove the k_i from the array, as we do not need it for the OLS
    for items in h_logit:
        del items[4]
        observed_choice.append(items[3])
        del items[3]
    logit_reg_prob = LogisticRegression().fit(h_logit, observed_choice)
    return [logit_reg_prob.intercept_,logit_reg_prob.coef_[0][0],logit_reg_prob.coef_[0][1],logit_reg_prob.coef_[0][2]]

def priceSOLVER(x,history,cost):
    betas_logit = probLOGIT(history)
    betas_ols = priceOLS(history)
    
    #import the regression Beta's from the results of the LOGIT and OLS regressions
    b_0 = betas_logit[0]
    b_x = betas_logit[1]
    b_own = betas_logit[2]
    b_opp = betas_logit[3]
    
    bo_0 = betas_ols[0]
    bo_x = betas_ols[1]
    bo_own = betas_ols[2]
    #print(b_0, b_x, b_own, b_opp)
    #print(bo_0, bo_x, bo_own)
    def rev(pric):
        chi = b_0+b_x*x+b_own*pric+b_opp*(bo_0 + bo_x*x+bo_own*pric) 
        return ((pric - cost)*(1-(1/(1+np.exp(chi)))))
    #print(b_0, b_x, b_own, b_opp)
    #print(bo_0, bo_x, bo_own)
    #print('b_0 = ', b_0, ' b_own = ', b_own,' b_x = ',  b_x)
    guess_duo = [[]]
    del guess_duo[0]
    
    price_guesses = np.arange(cost,max_price,solve_int)
    
    
                
    for guess in price_guesses:
        guess_duo.append([guess,rev(guess)])
    
    price = max(guess_duo, key=operator.itemgetter(1))[0]
   
   
   
    return (price)


#plots analytic price lines
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


################################################################################

#this function run several simulations and make barplots of interesting variables
def simulationrunner(confs,N,plots,bar_prof,bar_ms):
    barWidth = 0.25
    avg_pi_a = []
    avg_pi_b = []
    avg_CS = []
    ssd_pi_a = []
    ssd_pi_b = []
    ssd_CS = []
    
    avg_ms_a = []
    avg_ms_b = []
    avg_ms_o = []
    ssd_ms_a = []
    ssd_ms_b = []
    ssd_ms_o = []
    
    
    
    # run over the number of different configuration files
    for n in range(0,confs):
        configname = 'prdiconfig'+str(n)+ '.ini'
        array_pi_a = []
        array_pi_b = []
        array_CS = []
        
        array_ms_a = []
        array_ms_b = []
        array_ms_o = []
        # run over N simulations per configuration file
        for i in range(0,N):
            #add the results tuple of an inidividual simulation to a results list
            results.append(simulation(configname,plots))
    

        
        # add the entries of the result tuples to the lists that have all the 
        # profits per firm for a certain configuration in them for all N
        for item in results:
            array_pi_a.append(item[0])
            array_pi_b.append(item[1])
            array_CS.append(item[2])
            array_ms_a.append(item[3])
            array_ms_b.append(item[4])
            array_ms_o.append(item[5])
    
        # calculate the mean and std of these lists and add them to 
        # the final list which we use to plot
        avg_pi_a.append(sum(array_pi_a)/len(array_pi_a))
        avg_pi_b.append(sum(array_pi_b)/len(array_pi_b))
        avg_CS.append(sum(array_CS)/len(array_CS))
        
        avg_ms_a.append(sum(array_ms_a)/len(array_ms_a))
        avg_ms_b.append(sum(array_ms_b)/len(array_ms_b))
        avg_ms_o.append(sum(array_ms_o)/len(array_ms_o))
        
        #print('before CLEAR: ', array_pi_a,array_pi_b,array_CS)
        
        # std profits
        ssd_pi_a.append(np.std(array_pi_a))
        ssd_pi_b.append(np.std(array_pi_b))
        ssd_CS.append(np.std(array_CS))
        
        # std market share
        ssd_ms_a.append(np.std(array_ms_a))
        ssd_ms_b.append(np.std(array_ms_b))
        ssd_ms_o.append(np.std(array_ms_o))
        
        array_pi_a.clear()
        array_pi_b.clear()
        array_CS.clear()
        
        array_ms_a.clear()
        array_ms_b.clear()
        array_ms_o.clear()
        
        results.clear()
        
        #print('after CLEAR: ',array_pi_a,array_pi_b,array_CS)
        
        #print('STD  ', ssd_pi_a,ssd_pi_b,ssd_CS)
        
    r1 = np.arange(len(avg_pi_a))
    r2 = [x + barWidth for x in r1]
    r3 = [x + 2*barWidth for x in r1]
    
    
    #NUMBER AND TEXT IS MANUAL!!!!!!
    xticks_txt = [r'$0.1$',r'$0.5$',r'$1$',r'$2$',r'$5$',r'$10$',r'$100$',r'$250$',r'$350$']
    
    # Make the barplot for profits
    if bar_prof == True:
        plt.bar(r1, avg_pi_a, color='cornflowerblue', width=barWidth, edgecolor='white', label='Firm A',yerr=ssd_pi_a, align='center', alpha=0.5, ecolor='black', capsize=3)
        plt.bar(r2, avg_pi_b, color='slateblue', width=barWidth, edgecolor='white', label='Firm B',yerr=ssd_pi_b, align='center', alpha=0.5, ecolor='black', capsize=3)
        plt.bar(r3, avg_CS, color='mediumblue', width=barWidth, edgecolor='white', label='CS',yerr=ssd_CS, align='center', alpha=0.5, ecolor='black', capsize=3)
         
        # Add xticks on the middle of the group bars
        plt.xlabel(r'$\lambda$', fontweight='bold')
        #plt.xlabel(r'$T$', fontweight='bold')
        plt.ylabel('surplus per consumer', fontweight='bold')
        #plt.xticks([r + barWidth for r in range(len(avg_pi_a))], [r'$0.5$',r'$1$',r'$2$',r'$5$',r'$10$', r'$100$', r'$250$'])
        #plt.xticks([r + barWidth for r in range(len(avg_pi_a))], [r'$250$', r'$350$'])
        plt.xticks([r + barWidth for r in range(len(avg_pi_a))], xticks_txt)
        # Create legend & Show graphic
        plt.legend()
        plt.title(r'$\omega = $'+str(w)+'  ,  '+r'$c^B = $'+ str(cost_b)+'  ,  '+r'$T = $'+ str(T)+'  ,  '+r'$\theta= $'+ str(training))
        histname = 'Profit_'+'Ua' +str(util_a) + 'Ca'+str(cost_a)+'w'+ str(w)+'Cb'+str(cost_b)+'t'+str(tech)+'T'+str(T)+'tr'+str(training)+'lmd'+str(precision)
        plt.savefig(histname+ '.pdf')
        plt.show()
        
        
    # Make the barplot for market shares
    if bar_ms == True:
        plt.bar(r1, avg_ms_a, color='cornflowerblue', width=barWidth, edgecolor='white', label='Firm A',yerr=ssd_ms_a, align='center', alpha=0.5, ecolor='black', capsize=3)
        plt.bar(r2, avg_ms_b, color='slateblue', width=barWidth, edgecolor='white', label='Firm B',yerr=ssd_ms_b, align='center', alpha=0.5, ecolor='black', capsize=3)
        plt.bar(r3, avg_ms_o, color='mediumblue', width=barWidth, edgecolor='white', label='Outside Option',yerr=ssd_ms_o, align='center', alpha=0.5, ecolor='black', capsize=3)
         
        # Add xticks on the middle of the group bars
        plt.xlabel(r'$\lambda$', fontweight='bold')
        #plt.xlabel(r'$T$', fontweight='bold')
        plt.ylabel('market share', fontweight='bold')
        #plt.xticks([r + barWidth for r in range(len(avg_pi_a))], [r'$0.5$',r'$1$',r'$2$',r'$5$',r'$10$', r'$100$', r'$250$'])
        #plt.xticks([r + barWidth for r in range(len(avg_pi_a))], [r'$250$', r'$350$'])
        plt.xticks([r + barWidth for r in range(len(avg_ms_a))], xticks_txt)
        # Create legend & Show graphic
        plt.legend()
        plt.title(r'$\omega = $'+str(w)+'  ,  '+r'$c^B = $'+ str(cost_b)+'  ,  '+r'$T = $'+ str(T)+'  ,  '+r'$\theta= $'+ str(training))
        histname = 'MarketS_'+'Ua' +str(util_a) + 'Ca'+str(cost_a)+'w'+ str(w)+'Cb'+str(cost_b)+'t'+str(tech)+'T'+str(T)+'tr'+str(training)+'lmd'+str(precision)
        plt.savefig(histname+ '.pdf')
        plt.show()

#RUN STUFF HERE
        
#simulationrunner(#configs, #simulations, plots, barplot_profits, barplot_marketshare)
simulationrunner(2,1,True,False,False)


# note you need as many configfiles in your folder as #configs



