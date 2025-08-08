import numpy as np
from stockstream import PolygonData
import matplotlib.pyplot as plt

API_key = 'aSE_Q2Vmnh9BCPSumGUceCJphILlB3ag'
ticker = 'AAPL'
limit = 900000
save_dir = "./data/"
price_name = save_dir + ticker + "-min.npy"
volume_name = save_dir + ticker + "-min-volume.npy"

stream_obj = PolygonData(API_key)

#min_data = stream_obj.retrieve_data(ticker, limit)

min_data = np.load(price_name)[:200]
volume_data = np.load(volume_name)[:200]

stream_obj.volume = volume_data
stream_obj.output = min_data


vwap_data = stream_obj.vwap(min_data, 18)
vwap_boot = stream_obj.vwap_boot(min_data, 18)

xdata = range(len(min_data))


plt.plot(xdata, min_data, label='1min Close Price')
plt.plot(xdata, vwap_data, label='18min VWAP')
plt.plot(xdata, vwap_boot, label='18min Bootstrapped VWAP', color="green")

plt.show()

