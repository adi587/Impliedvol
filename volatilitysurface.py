import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt
from Optionsimpliedvol import ImpliedVolatility
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score



def clean_data(df):
    date_now=dt.datetime.now()
    maturity_dates=[]
    price=[]
    for c in range(len(df)):
        
        price=np.append(price,(float(df['Bid'][c])+float(df['Ask'][c]))/2)
        
        name=df['Contract Name'][c]
        d=name[len(ticker):(len(ticker)+6)]
        maturity_date=dt.datetime(2000+int(d[0:2]),int(d[2:4]),int(d[4:6]))
       
        time_to_maturity=maturity_date-date_now
        
        time_to_maturity=time_to_maturity.total_seconds()
        time_to_maturity=time_to_maturity/86400
        
        time_to_maturity=time_to_maturity/365
        
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
       
        imp=model.get_impliedvol([strike[i]], [df['Price'][i]])
        print(strike[-1],imp) #To visually see each implied vol
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
    print("R2 test set: "+str(r2_score(y_test, y_pred)))
    y_pred=lin_reg.predict(poly_reg.fit_transform(x_train))
    print("R2 train set: "+str(r2_score(y_train, y_pred)))
    return lin_reg
def plot_3D_model(ticker,lin_reg,t_start,t_end,strike_start,strike_end,resolution):
    strikeline=np.outer(np.linspace(strike_start, strike_end,resolution),np.ones(resolution))
    
    timeline=np.outer(np.linspace(t_start,t_end,resolution),np.ones(resolution)).T
    a_0=lin_reg.intercept_
    a_1=lin_reg.coef_[0][1]
    a_2=lin_reg.coef_[0][2]
    a_3=lin_reg.coef_[0][3]
    a_4=lin_reg.coef_[0][4]
    a_5=lin_reg.coef_[0][5]
    impvolline=a_0+a_1*strikeline+a_2*timeline+a_3*strikeline**2+a_4*strikeline*timeline+a_5*timeline**2

    
    ax=plt.axes(projection='3d')
    ax.plot_surface(strikeline, timeline, impvolline)
    ax.set_xlabel('Strikes (USD)')
    ax.set_ylabel('Time (yrs)')
    ax.set_zlabel('IV')
    plt.title("IV surface for "+ticker)
    plt.show()
    
    return[strikeline,timeline,impvolline]

