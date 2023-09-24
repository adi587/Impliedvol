# -*- coding: utf-8 -*-
"""
Created on Tue Aug 30 16:06:35 2022

@author: adithya
"""
from Stockoptions import Stockoption
import math
import numpy as np

class Binomialoption(Stockoption):
    '''
    Stock option class has all the parameters defined for underlying asset
    Attributes:
        S_0=curent price of underlying
        c=strike price
        r=risk free rate
        t_0=time to expiration
        N=no.steps in binomial tree
        g=gain in logarithim of price per time step
        u/d=the actual gain/loss in price per time step
        prob=probability of price increasing in a given time step
        Usage eg: option=Binomialoption(S_0=30, c=40, r=0.035, t_0=2, N=250,
                              params={"sigma":0.5,
                                      "is_eu":True,
                                      "is_call":True})
        
    Methods:
        underly_tree_initial= Creates the tree of the underlying assets price with time
        america_excersise=When american option checks if early excersise better
        valuation_tree=Creates the tree of the options value with time
        price=Gives the current price of option
        greeks=Calculates/estimates the greeks through the binomial model
                and returns the values in a dictionary.
    '''
    
    def __init__(self,S_0,c,r,t_0,N,params):
        super().__init__(S_0,c,r,t_0,N,params)
        self.g=self.sigma*math.sqrt(self.dt)
        self.u=math.exp(self.g)
        self.d=math.exp(-self.g)
        self.prob=(math.exp((self.r-self.div)*self.dt)-
                   self.d)/(self.u-self.d)
        
        

    def underly_tree_initial(self):
        self.underly_tree=[np.array([float(self.S_0)])]
        for i in range(self.N):
            recent=self.underly_tree[-1]
            add=np.concatenate(([recent[0]*self.d],recent*self.u,))
            self.underly_tree.append(add)
            
    def america_excerise(self,i,j):
        if self.is_call: early=(self.underly_tree[i][j]-self.c)
        else: early=(self.c-self.underly_tree[i][j])
        

        return max(early,self.val_tree[i][j])

            
    def valuation_tree(self):
        self.underly_tree_initial()       
        self.val_tree=[self.underly_tree[i].copy() 
                           for i in range(len(self.underly_tree))]

        for i in range(len(self.val_tree[-1])):
            if self.is_call: 
                self.val_tree[-1][i]=max(self.underly_tree[-1][i]-self.c,0)
            else: 
                self.val_tree[-1][i]=max(self.c-self.underly_tree[-1][i],0)
            
        
        for i in reversed(range(self.N)):           
            for j in range(len(self.val_tree[i])):
                self.val_tree[i][j]=max((self.prob*self.val_tree[i+1][j+1]+
                                     (1-self.prob)*
                                     self.val_tree[i+1][j])*self.df,0)
                if not (self.is_european):
                    self.val_tree[i][j]=self.america_excerise(i, j) 

                    
    def price(self):
        self.valuation_tree()
        return self.val_tree[0][0]    
    
    def greeks(self):
        self.valuation_tree()
        self.underly_greeks=[self.underly_tree[i].copy() 
                           for i in range(len(self.underly_tree))]        
        self.val_greeks=[self.val_tree[i].copy() 
                           for i in range(len(self.val_tree))] 
        
        for i in reversed(range(self.N)):
            self.underly_greeks[i]=np.append(self.underly_greeks[i],
                                             self.S_0*self.u**(i+1)/self.d)
            self.underly_greeks[i]=np.insert(self.underly_greeks[i],0,
                                             self.S_0*self.d**(i+1)/self.u)
            
            
            if self.is_call: 
                self.val_greeks[i]=np.append(self.val_greeks[i],
                                             max(self.underly_greeks[i][-1]-self.c,0))
                self.val_greeks[i]=np.insert(self.val_greeks[i],0,
                                             max(self.underly_greeks[i][0]-self.c,0))
            else: 
                self.val_greeks[i]=np.append(self.val_greeks[i],
                                             max(self.c-self.underly_greeks[i][-1],0))
                self.val_greeks[i]=np.insert(self.val_greeks[i],0,
                                             max(self.c-self.underly_greeks[i][0],0))
            
            
            self.val_greeks[i][-1]=max((self.prob*self.val_greeks[i+1][-1]+
                                     (1-self.prob)*
                                     self.val_greeks[i+1][-2])*self.df,0)
            self.val_greeks[i][0]=max((self.prob*self.val_greeks[i+1][1]+
                         (1-self.prob)*
                         self.val_greeks[i+1][0])*self.df,0) 
            
            if not (self.is_european):
                if self.is_call: early=(self.underly_greeks[i][-1]-self.c)
                else: early=(self.c-self.underly_greeks[i][-1])
                self.val_greeks[i][-1]=max(early,self.val_greeks[i][-1]) 
                if self.is_call: early=(self.underly_greeks[i][0]-self.c)
                else: early=(self.c-self.underly_greeks[i][0])
                self.val_greeks[i][0]=max(early,self.val_greeks[i][0])
                

        #delta                
        self.delta=(self.val_greeks[0][2]-
                   self.val_greeks[0][0])/(self.underly_greeks[0][2]-
                                         self.underly_greeks[0][0])                  

        #gamma                                           
        del_u=(self.val_greeks[0][2]-
                self.val_greeks[0][1])/(self.underly_greeks[0][2]-
                                        self.underly_greeks[0][1])              
        del_d=(self.val_greeks[0][1]-
                self.val_greeks[0][0])/(self.underly_greeks[0][1]-
                                        self.underly_greeks[0][0])    
        und_u=(self.S_0+self.underly_greeks[0][2])/2                                         
        und_d=(self.S_0+self.underly_greeks[0][0])/2
        
        self.gamma=(del_u-del_d)/(und_u-und_d)
        
        #theta
        thetaroot=Binomialoption(self.S_0, self.c, self.r, self.t_0-self.dt, self.N,
                            {"sigma":self.sigma,
                             "is_eu":self.is_european,
                             "is_call":self.is_call})
        self.theta=(thetaroot.price()-self.val_tree[0][0])/(self.dt)
        
        #vega
        vegaroot=Binomialoption(self.S_0, self.c, self.r, self.t_0, self.N,
                            {"sigma":self.sigma*1.001,
                             "is_eu":self.is_european,
                             "is_call":self.is_call})
        self.vega=(vegaroot.price()-self.val_tree[0][0])/(0.001*self.sigma)
        
        #rho       
        rhoroot=Binomialoption(self.S_0, self.c, self.r*1.001, self.t_0, self.N,
                            {"sigma":self.sigma,
                             "is_eu":self.is_european,
                             "is_call":self.is_call})
        self.rho=(rhoroot.price()-self.val_tree[0][0])/(0.001*self.r)
        
        return {"Delta":self.delta,
                "Gamma":self.gamma,
                "Theta":self.theta,
                "Vega":self.vega,
                "Rho":self.rho}
        
        
        
'''
option=Binomialoption(S_0=65, c=50, r=0.035, t_0=1, N=250,
                      params={"sigma":0.5,
                              "is_eu":True,
                              "is_call":True})

print(option.price())
print(option.greeks())
'''