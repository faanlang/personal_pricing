import configparser
from itertools import product 

#This program writes one or more config files to be used in various simulation 
#programs of price discrimination markets

#for each parameter list all values that you want in a list
#example cost_b_list = [0.5, 1] and w_list = [0.55, 1] , all other lists having  1 entry
#will produce 4 configuration files with all combinations of different parameters

#configfiles are produced numbered

#note w = \omega
#note tech = t
#note precision = \lambda
#note T = runtime AFTER training (=\theta)

#FILL IN PARAMETER VALUES HERE
util_a_list = [2] 
cost_a_list = [1] 
cost_b_list = [1] 
w_list = [0.8] 
tech_list = [0.5] 
precision_list = [10] 
T_list = [1000] 
training_list = [1000] 

#Now make a product of all parameter combinations
res = list(product(util_a_list,cost_a_list,cost_b_list,w_list
                   ,tech_list,precision_list,T_list,training_list))

#print this for quick check reference
print(res)


###############################################################################

#write 1 combination in 1 file
def configwriter(util_a, cost_a, cost_b, w, tech, precision, T, training, number):
    
    
    util_a_str = str(util_a)
    cost_a_str = str(cost_a)
    cost_b_str = str(cost_b)
    w_str = str(w)
    tech_str = str(tech)
    precision_str = str(precision)
    T_str = str(T)
    training_str = str(training)    
    
    number_str = str(number)
    config = configparser.ConfigParser()
    config['parameters'] = {'util_a': util_a_str,
                    'cost_a': cost_a_str,
                    'cost_b': cost_b_str,
                    'w': w_str,
                    'tech': tech_str,
                    'precision': precision_str,
                    'T': T_str,
                   'training': training_str}

    #Name of the produced configfile
    configname = 'prdiconfig' + number_str + '.ini'
    with open(configname, 'w') as configfile:
        config.write(configfile)

###############################################################################

#repeat for all combinations
def configfiller():
    i = 0
    for item in res:
        arguments = item
        configwriter(*arguments,i)
        i +=1

###############################################################################

#run the function
configfiller()