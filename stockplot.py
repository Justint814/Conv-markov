import numpy as np
from stockstream import PolygonData
import matplotlib.pyplot as plt

API_key = 'aSE_Q2Vmnh9BCPSumGUceCJphILlB3ag'
ticker = 'AAPL'
limit = 700

stream_obj = PolygonData(API_key)

min_data = stream_obj.retrieve_data(ticker, limit)
ema_data = stream_obj.vwap(min_data, 18)

xdata = range(len(min_data))

plt.plot(xdata, min_data, label='1min Close Price')
plt.plot(xdata, ema_data, label='18min ema')

plt.show()