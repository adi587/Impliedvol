# -*- coding: utf-8 -*-
"""
Created on Sat Sep  3 23:55:49 2022

@author: adithya
"""
from Stockoptions import Stockoption
import math
import numpy as np
from Binomialoption import Binomialoption
from scipy.stats import norm

class ImpliedVolatility:
    def __init__(self,S_0,r,t_0,div,N,res,call,eu,binomial):
        self.S_0=S_0 #stock price now
        self.r=r     #riskfree rate
        self.t_0=t_0 #time to expiry
        self.N=N     #no. steps in the tree
        self.div=div #divedends of underlying
        self.is_call = call.get("is_call", True)  
        self.is_eu=eu.get("is_eu",True)
        self.res=res #The resolution that volatility will be correct to
        self.binomial=binomial.get("binomial",False) # To choose via binomial or not
        
    def optionval(self,c,sigma):
        opt=Binomialoption(self.S_0, c, self.r, self.t_0, self.N,{
                    "div":self.div,
                    "sigma":sigma,
                    "is_call":self.is_call,
                    "is_eu":self.is_eu})        
         
        return opt.price()   
    
    
    
    def black(self,c,sigma):
        d=(math.log(self.S_0/c)+(self.r+(sigma**2)/2)*self.t_0)/(sigma*math.sqrt(self.t_0))
        ans=self.S_0*norm.cdf(d)-c*math.exp(-self.r*self.t_0)*norm.cdf(d-sigma*math.sqrt(self.t_0))
        return ans

    def get_impliedvol(self,strikes,opt_values):
        impvol=[]
        for i in range(len(strikes)):
            #Bisection calc:
            a=0.1
            b=1
            if self.binomial:
                while not self.optionval(strikes[i], a)<opt_values[i]:
                    a=a/2
                while not self.optionval(strikes[i], b)>opt_values[i]:
                    b=b*2
            else:
                while not self.black(strikes[i],a)<opt_values[i]:
                    a=a/2
                while not self.black(strikes[i],b)>opt_values[i]:
                    b=b*2     
            d=(a+b)/2
            impv=self.optionval(strikes[i], d)
            
            
            while abs(impv-opt_values[i])>self.res:
                if impv>opt_values[i]:
                    b=d
                else: a=d
                d=(a+b)/2
                if self.binomial:
                    impv=self.optionval(strikes[i], d)
                else:
                    impv=self.black(strikes[i], d)
            
                
            impvol=np.append(impvol,d)    
                                                      
        return impvol
    
'''           

model=ImpliedVolatility(151.76, 0.04, 0.0465753,0.0061, 250,0.001,
                        {"is_call":True},{"is_eu":True},{'binomial':True})

print(model.get_impliedvol([138], [15.575]))



'''

#model=ImpliedVolatility(130.54, 0.05, 1/365,0, 250,0.001,
#                        {"is_call":True},{"is_eu":True},{'binomial':True})

#print(model.get_impliedvol([133], [0.385]))