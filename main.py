from volatilitysurface import clean_data,implied_vol_calc,train_model,plot_3D_model
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

ticker='AMZN'
current_price=129.12# #current stock price value
risk_free=0.05 #risk free rate
div=0. #dividends per year

df=pd.read_table('AMZNoptionprice.tsv')


df=clean_data(df)
df=implied_vol_calc(df)
lin_reg=train_model(df)
results=plot_3D_model(ticker,lin_reg,0.05,0.25,65,190,200)

strikesplot=[row[50] for row in results[0]]
IVplot=[row[50] for row in results[2]]
plt.plot(strikesplot,IVplot)
plt.ylim([0.25,0.9])
plt.xlim([65,190])
plt.plot([current_price]*len(IVplot),np.linspace(0,1,len(IVplot)),'--')
plt.title("At 1 month expiry "+ticker)
plt.xlabel("Strikes (USD)")
plt.ylabel("IV (annual)")
plt.legend(["IV","Current price"])
plt.grid()
plt.show()


timeplot=results[1][0]
IVplot=results[2][106]
plt.plot(timeplot,IVplot)
plt.xlim([0.05,0.25])
plt.xlabel("Time (yrs)")
plt.ylabel("IV (annual)")
plt.title("At 133 strike "+ticker)
plt.grid()
plt.show()
