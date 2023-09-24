# -*- coding: utf-8 -*-
"""
Created on Tue Aug 30 15:51:40 2022

@author: adithya
"""
import math

class Stockoption:
    def __init__(self,S_0,c,r,t_0,N,params):
        self.S_0=S_0 #stock price now
        self.c=c     #strike price
        self.r=r     #riskfree rate
        self.t_0=t_0 #time to expiry
        self.N=N     #no. steps in the tree
        
        self.div = params.get("div", 0) # Dividend yield
        self.sigma = params.get("sigma", 0) # Volatility
        self.is_call = params.get("is_call", True) # Call or put
        self.is_european = params.get("is_eu", True) # Eu or Am
        
        self.dt=t_0/float(N)    #time step
        self.df = math.exp(
            -(r-self.div) * self.dt) # Discount factor
        
        