# -*- coding: utf-8 -*-
"""
Created on Tue Sep 27 21:27:34 2022

@author: adithya
"""

import math
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import datetime as dt
from Optionsimpliedvol import ImpliedVolatility
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

#maturity_dates=['2022-10-14','2022-10-21','2022-10-28','2022-11-04']

ticker='AMZN'
current_price=129.12#114.41 #current stock price value
risk_free=0.05 #risk free rate
div=0. #divedends per year

df=pd.read_table('AMZNoptionprices200922.tsv')

#date_now=dt.datetime.now()
#maturity_dates=[]
#price=[]
def clean_data(df):
    date_now=dt.datetime.now()
    maturity_dates=[]
    price=[]
    for c in range(len(df)):
        
        price=np.append(price,(float(df['Bid'][c])+float(df['Ask'][c]))/2)
        
        name=df['Contract Name'][c]
        d=name[len(ticker):(len(ticker)+6)]
        maturity_date=dt.datetime(2000+int(d[0:2]),int(d[2:4]),int(d[4:6]))
        #maturity_date=dt.datetime(2023,10,13)
        
        LTD=date_now
        #date=dt.datetime(int(LTD[0:4]),int(LTD[5:7]),int(LTD[8:10]))
        time_to_maturity=maturity_date-date_now
        
        time_to_maturity=time_to_maturity.total_seconds()
        time_to_maturity=time_to_maturity/86400
        #print(time_to_maturity)
        time_to_maturity=time_to_maturity/365
        #time_to_maturity=1/365
        maturity_dates=np.append(maturity_dates,time_to_maturity)
    
    df['Yrs to maturity']=maturity_dates
    df['Price']=price
   
    return df

def implied_vol_calc(df):
    implied_vol=[]
    strike=[]
    for i in range(len(df)):
        
        date=df['Yrs to maturity'][i]
        
        
        model=ImpliedVolatility(current_price, risk_free,date ,div, 250,0.001,
                                {"is_call":True},{"is_eu":True},{'binomial':True})
        strike=np.append(strike,int(df['Strike'][i]))
        #print(date,current_price, risk_free,date ,div,[strike[i]], [price[i]])
        imp=model.get_impliedvol([strike[i]], [df['Price'][i]])
        print(strike[-1],imp)
        implied_vol=np.append(implied_vol,imp)
    df['implied vol']=implied_vol
    return df

def train_model(df):
    x_train, x_test, y_train, y_test = train_test_split(df[['Strike','Yrs to maturity']], df[['implied vol']],
                                                        test_size = 0.2,
                                                        random_state = 1)

    poly_reg=PolynomialFeatures(degree=2)
    x_poly=poly_reg.fit_transform(x_train)

    lin_reg=LinearRegression()
    lin_reg.fit(x_poly,y_train)

    y_pred = lin_reg.predict(poly_reg.fit_transform(x_test))
    coefficient_of_dermination=r2_score(y_test, y_pred)
    print("Test set: "+str(coefficient_of_dermination))
    y_pred1=lin_reg.predict(poly_reg.fit_transform(x_train))
    print("Train set: "+str(r2_score(y_train, y_pred1)))
    return [lin_reg,coefficient_of_dermination]
def plot_3D_model(lin_reg,t_start,t_end,strike_start,strike_end,resolution):
    strikeline=np.outer(np.linspace(strike_start, strike_end,resolution),np.ones(resolution))
    
    timeline=np.outer(np.linspace(t_start,t_end,resolution),np.ones(resolution)).T
    a_0=lin_reg.intercept_
    print(lin_reg.coef_)
    a_1=lin_reg.coef_[0][1]
    a_2=lin_reg.coef_[0][2]
    a_3=lin_reg.coef_[0][3]
    a_4=lin_reg.coef_[0][4]
    a_5=lin_reg.coef_[0][5]
    impvolline=a_0+a_1*strikeline+a_2*timeline+a_3*strikeline**2+a_4*strikeline*timeline+a_5*timeline**2

    fig=plt.figure()
    ax=plt.axes(projection='3d')
    ax.plot_surface(strikeline, timeline, impvolline)
    ax.set_xlabel('Strikes (USD)')
    ax.set_ylabel('Time (yrs)')
    ax.set_zlabel('IV')
    plt.show()
    return[strikeline,timeline,impvolline]

df=clean_data(df)
df=implied_vol_calc(df)
lin_reg=train_model(df)[0]
results=plot_3D_model(lin_reg,0.05,0.25,65,190,200)

tester=[row[50] for row in results[0]]
tester1=[row[50] for row in results[2]]
plt.plot(tester,tester1)
#plt.xlim([125,145])
plt.ylim([0.25,0.9])
plt.xlim([65,190])
#plt.plot(tester,[0.55]*len(tester))
#plt.plot([current_price*1.1]*len(tester1),np.linspace(0,1,len(tester1)))
plt.plot([current_price]*len(tester1),np.linspace(0,1,len(tester1)),'--')
plt.title("At 1 month expiry")
plt.xlabel("Strikes (USD)")
plt.ylabel("IV (annual)")
plt.legend(["IV","Current price"])
plt.grid()
plt.show()

timeplot=results[1][0]
IVplot=results[2][106]
plt.plot(timeplot,IVplot)
plt.xlabel("Time (yrs)")
plt.ylabel("IV (annual)")
plt.title("At 133 strike")
plt.grid()
plt.show()

