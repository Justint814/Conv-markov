import numpy as np
from stockstream import PolygonData
import matplotlib.pyplot as plt

API_key = 'aSE_Q2Vmnh9BCPSumGUceCJphILlB3ag'
ticker = 'AAPL'
limit = 900
save_dir = "./data/"
price_name = save_dir + ticker + "-min.npy"
volume_name = save_dir + ticker + "-min-volume.npy"

stream_obj = PolygonData(API_key)

min_data = stream_obj.retrieve_data(ticker, limit)
volume_data = stream_obj.volume


#min_data = np.load(price_name)[0:2000:1]
#volume_data = np.load(volume_name)[0:2000:1]

stream_obj.volume = volume_data
stream_obj.output = min_data

vwap_data = stream_obj.vwap(min_data, 200)
ema_data = stream_obj.ema(min_data, 14)
vwap_boot = stream_obj.vwap(ema_data, 20)
vwap_boot2= stream_obj.vwap(vwap_boot, 80)

ema2 = stream_obj.ema(min_data, 12)
ema3 = stream_obj.ema(min_data, 9)

xdata = range(len(min_data))


plt.plot(xdata, min_data, label='1min Close Price')
#plt.plot(xdata, ema_data, label='18min VWAP')
plt.plot(xdata, vwap_boot, label='18min Bootstrapped VWAP', color="red")
plt.plot(xdata, vwap_boot2)
plt.plot(xdata, ema3, color="blue")
#plt.plot(xdata, vwap_minus2)

plt.show()

#if most recent vwap extrema from last transition of vwaps is a certain percentage of price, enter at vwap crossings and sell at vwap extrema

