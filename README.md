# Impliedvol
After learning about the shortcomings of the Black Scholes (BS) model I wanted to see how implied volatility (IV) varies for different equity options (wanted to compare with FX options to but couldnt access this data easily). To do this I use the same files from my previous Binomialoption repository, to calculate implied volatiltiy given an option strike and price.
# Volatility surface model and training
As there are only a discrete select set of option strikes traded on the markets I need to create a model that can predict option prices for all strikes and maturity dates given from market data. For the model I draw from the Dumas, Fleming and Whaley 1996 paper were they used: $IV=a_0+a_1K+a_2T+a_3K^2+a_4KT+a_5T^2$ were $IV,K,T$ is implied volatility, strike and time to maturity respectively. The $a_i$ coefficients are to be determined from training the model on historical data. This model is useful as the quadratic nature will capture the curving nature of IV without overfitting (which happen with including higher order polynomials). The market data to train the model was downloaded online for various strikes and expiration dates (setting up an API to dynamically retrieve market information would be too time consuming for the purpose of this project). The model was trained using a linear regressor (a mean squared distance loss function) after transforming the features into polynomial features. 
# Analysis
I first constructed volatility surfaces for Amazon (AMZN) options. The price of the stock at the time of training was 129.12 USD. I set the risk free rate to 0.05 matching treasury bill yields. The dividends rate was set to 0 (An online search shows amazons low dividends rate). The risk free rate and dividends do not same to have much effect on the IV values (varying both from 0 to 0.1 only changed IV by 0.03 at most) and shape so I wasn't worried about using exact figures. I only use call options as by the put-call parity the shape of the IV curves should be similiar. I selected a range of strikes and 5 different maturity dates from 0.05yrs to 0.25yrs (A total of 170 data points). Heavily ITM and OTM strikes were not included as these were seen to skew the model significantly. To train the model I split the data with 0.8 as the training set and 0.2 for testing. I then used a k-fold cross validation on the training set to determine how well suited the model is to the data. 5 splits were taken and the average $R^2$ score was 0.8. This seemed largely due to 1 split giving an anomalously low score of 0.62 while the rest were above 0.8, so the model seems well suited to this data. The $R^2$ coefficient on the entire training set was 0.84 and for the test set it was 0.89. This is a strong coefficient of determination for the test set which may be due to the model being trained on more data (the entire training set rather than a subset as in the cross validation) and hence being a better predictor of IV. However I think the more likely explanaiton is that the test data coincidentally lined up well with the model as this value exceeds all those in the cross validation and its own training set. None the less we can still conclude this model does a good job at predicting the IV values. I then plotted the surface for a range of points giving this 3D plot:

![image](https://github.com/adi587/Volatilitysurfaces/assets/63116085/f3140a08-1be4-4111-a7b7-464bbff798da)

Note that the z-axis is the implied volatility. A clear curvature can be seen between the strike price and implied volatility. Also a less pronounced curve can be seen with the implied volatiltiy and time to maturity. To see these effects clearer, IV is plotted against strike at 1 month time to maturity and IV plotted against time at 133 strike.

# IV dependance on strike (AMZN)

![image](https://github.com/adi587/Volatilitysurfaces/assets/63116085/394cc616-9d0d-4cdc-9f61-8e43965ab747)

First on the IV and strike relation, the orange dotted line is to show the stock price at time of training. Before the 1987 crash, equity options like these did not exhibit volatility skews. The standard answer for volatility smiles this type of dependance between IV and strike is to use short-comings of the Black-Scholes model. Black-Scholes assumes the underlying assets price to move as a lognormal distribution. This underpredicts the probability of larger price movements (tails of distribution) and so underprices low and high strike options. Therefore the market prices these 'tail' options higher than what Black-Scholes predicts, giving a higher IV for strikes at the tails. The IV in this case can be seen to be assymetric (IV is larger for lower strikes compared to higher strikes) which means (using the above reasoning) that there is a higher probability of a large downturn rather than an upturn in price. This may be true as there are more ways for a business to rapidly lose value than increase value however I'm unsure if this is the main factor (should probably do some better analysis on this later). Alternatively the skew can come from a higher demand in low strikes driving option prices higher. 

I spoke to a senior quant trader about this problem. He theorised the skew arises from insurance on positions in the underlying asset. For example if a firm is long on amazon they may buy heavily OTM put options at a low strike to hedge against large downfalls in the stocks price. This leads to an increased demand for low OTM puts driving the prices and IV up. Through put-call parity this also drives IV of call options up. The same arguement can be used for a firm that is short amazon. They may buy heavily OTM calls to hedge against an upturn in the market and following the same logic as previously, this gives an increased IV at higher strikes. Finally the market is usually overall long rather than short so there is a higher demand for the first type of insurance explained rather than the second. This causes the lower strikes to have higher IV than higher strikes giving the volatility skew we see (according to this argument the amount the volatility skews should give us an indication of market sentiment of the underlying). This argument would also explain why the skew arised after 87, as traders became more risk averse and hedged there positions against large swings in price. 

The final distribution seen is likely a superposition of both these effects (and probably several other effects not mentioned here). A full statistical analysis of price movements of the underlying could be done to see how the underlying price distribution compares to the log-normal distribution. From there we can attempt to quantify how much this effect impacts the IV distribution (i.e if the price distribution is very clsoe to log-normal then we know the driving factor of the shape of IV is due to another effect rather than large price swings). I might try do this in another project down the line. 

# IV dependance on time to maturirty (AMZN)

![image](https://github.com/adi587/Volatilitysurfaces/assets/63116085/41155d06-ea1d-46e6-9634-367d2586522f)

The time used in training set was from 0.05 to 0.25 yrs so our model is well suited to predict this range of maturity times. Looking at this graph we see IV initially increasing then decreasing with longer time to maturity. Downward facing curves is the market predicting that realised volatility (actual volatility of the underlying) will decrease in future. Upward facing curves predict the opposite (similiar to bonds term structure). We can compare the minimum of IV with historical volatiltiy and see that it is large(20 day average of historical volatiltiy is 0.26 compared to 0.34). Therefore the market is predicting an increase in volatiltiy in the next 1-2 months followed by a decrease. Volatlity is observed to be mean reverting so this increase will be followed by a decrease as it reverts back to the mean.

I also looked at AAPL which showed interesting structure in the volatiltiy surface. I will add this later once I figure out an explanation for this.
