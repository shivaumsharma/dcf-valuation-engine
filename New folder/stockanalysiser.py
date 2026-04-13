#to build an analysier wherre i can put buy hold or sell 


import yfinance as yf
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
import streamlit as st

def get_stock_data(ticker):
    stock=yf.Ticker(ticker)
    data=stock.history(period="1y")
    return data 

def add_indicators(data):
    data['MA50']= data['Close'].rolling(50).mean()
    data['MA200']= data['Close'].rolling(200).mean()

    delta=data['Close'].diff()
    gain=delta.clip(lower=0).rolling(14).mean()
    loss=(-delta.clip(upper=0)).rolling(14).mean()
    rs=gain/loss
    data['RSI']=100-(100/(1+rs))

    return data 


def generate_signals(data):
    latest=data.iloc[-1]
    buy=0
    sell=0

    if latest['Close']>latest['MA50']:
        buy+=1
    else:
        sell+=1
    
    if latest['MA50']>latest['MA200']:
        buy+=1
    else:
        sell+=1


    if latest['RSI']<30:
        buy+=1
    elif latest['RSI']>70:
        sell+=1

    if buy >=2:
        return "BUY"
    elif sell>=2:
        return "SELL"
    else:
        return "Hold"

def generate_signal_row(row):
    buy = 0
    sell = 0

    if row['Close'] > row['MA50']:
        buy += 1
    else:
        sell+=1

    if row['MA50'] > row['MA200']:
        buy += 1
   

    if row['RSI'] < 40:
        buy += 1
    elif row['RSI'] > 60:
        sell += 1

    if buy >= 2:
        return 1
    elif sell >= 2:
        return -1
    else:
        return 0
    


data=get_stock_data("INFY.NS")
data =add_indicators(data)
data = data.dropna()
data['Signal'] = data.apply(generate_signal_row, axis=1)
print(data[['Close', 'MA50', 'MA200', 'RSI', 'Signal']].tail())
# Calculate daily returns
data['Returns'] = data['Close'].pct_change()

# Strategy returns (use previous day's signal)
data['Strategy_Returns'] = data['Returns'] * data['Signal'].shift(1)

# Cumulative returns
data['Cumulative_Market'] = (1 + data['Returns']).cumprod()
data['Cumulative_Strategy'] = (1 + data['Strategy_Returns']).cumprod()


plt.figure(figsize=(10,5))
plt.plot(data['Cumulative_Market'], label='Market')
plt.plot(data['Cumulative_Strategy'], label='Strategy')
plt.legend()
plt.show()