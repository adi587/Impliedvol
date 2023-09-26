# Impliedvol
After learning about the shortcomings of the Black Scholes (BS) model I wanted to see how implied volatility varies between equities and fx. To do this I use the same files from my previous Binomialoption repository, specifically to calculate implied volatiltiy given option strikes and prices.
# Volatility surface model and training
As there are only a discrete select set of option strikes traded on the markets I need to create a model that can predict option prices for all strikes and maturity dates given market data. For the model I draw from (Dumas, Fleming and Whaley 1996) paper: $IV=a_0+a_1K+a_2T+a_3K^2+a_4KT+a_5T^2$ where $IV,K,T$ is implied volatility, strike and time to maturity respectively. The $a_i$ coefficients are to be determined from training the model on historical data. This model is useful as the quadratic nature will capture the 'smiling' nature of IV without overfitting (as would be seen with higher order polynomials). The market data to train the model was downloaded online for various strikes and expiration dates (setting up an API to dynamically retrieve market information would be unnecessarily time consuming for the purpose of this project). The model was trained using a linear regressor (mean squared distance loss function) after transforming the features into polynomial features. 
# Analysis
writing up
