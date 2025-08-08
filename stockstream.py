import numpy as np
import requests
import datetime
from alpaca.data.live import StockDataStream
import pandas as pd
import asyncio

'''
if __name__ == "__main__":
    api_key = 'aSE_Q2Vmnh9BCPSumGUceCJphILlB3ag'
    ticker = 'AAPL'
    limit = 20000
    today = datetime.date.today()
    from_date = today - datetime.timedelta(days=30)
'''

#Class for retrieving historical data from polygon.io API
class PolygonData:

    #Dictionary to store type of data retrieved from json response of the polygon API request.
    polydat = {"close": "c", "open": "p", "low": "l", "high": "h", "time": "t", "volume": "v", "vwap": "vw"}

    def __init__(self, API_key: str, days=600):
        self.api_key = API_key
        self.today = datetime.date.today()
        self.from_date = self.today - datetime.timedelta(days=days)
    
    def retrieve_data(self, ticker: str, num_points: int, timeframe='minute', data_type="close"):
        url: str = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/{timeframe}/{self.from_date}/{self.today}?limit={num_points}&apiKey={self.api_key}"
        response = requests.get(url)
        data = response.json()
        target_param = self.polydat[data_type]

        self.high = np.array([item["h"] for item in data.get('results', [])])
        self.low = np.array([item["l"] for item in data.get('results', [])])
        self.open = np.array([item for item in data.get('results', [])])
        self.close = np.array([item["c"] for item in data.get('results', [])])
        self.volume = np.array([item["v"] for item in data.get('results', [])])
        self.output = np.array([item[target_param] for item in data.get('results', [])])

        return self.output
    
    def ema(self, input, period):
        self.ema_arr = np.zeros_like(input)
        multiplier = 2 / (period + 1)

        #Initialize first EMA value as SMA
        ema_val = np.sum(input[0:period]) / period
        self.ema_arr[0:period] = [ema_val for i in range(period)]

        for i in np.arange(period, len(input), 1):
            ema_val = (input[i] * multiplier) + ema_val * (1 - multiplier)
            self.ema_arr[i] = ema_val

        return self.ema_arr

    def vwap(self, input, period):
        self.vwap_arr = np.zeros_like(input)

        #Initialize starting values in first period
        self.vwap_arr[0:period] = np.sum(self.volume[0:period] * input[0:period]) / np.sum(self.volume[0:period])

        for i in np.arange(period, len(input), 1):
            self.vwap_arr[i] = np.sum(self.volume[i - period:i] * input[i - period:i]) / np.sum(self.volume[i - period:i])

        return self.vwap_arr
    
    def vwap_boot(self, input, period, sample_size=2000):
        self.vwap_boot_arr = np.zeros_like(input)

       #Initialize starting values in first period
        self.vwap_boot_arr[0:period] = np.sum(self.volume[0:period] * input[0:period]) / np.sum(self.volume[0:period])

        for i in np.arange(period, len(input), 1):
            volume_init = self.volume[i - period:i]
            price_init = input[i - period:i]

            #Apply bootstrapping to data range
            volume_resample = np.random.choice(volume_init, size=sample_size, replace=True)
            price_resample = np.random.choice(price_init, size=sample_size, replace=True)

            self.vwap_boot_arr[i] = np.sum(volume_resample * price_resample) / np.sum(volume_resample)

        return self.vwap_boot_arr

