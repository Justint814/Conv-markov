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

    def __init__(self, API_key: str, days=40):
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
    
    def boot_sma(self, input, period, sample_size=200):
        sma = np.zeros_like(input)
        std_dev = np.zeros_like(input)

        #Initialize starting values in first period
        sma[0:period] = np.sum(input[0:period]) / len(input[0:period])

        for i in np.arange(period, len(input), 1):
            sma[i] = np.sum(input[i-period:i]) / period
            std_dev[i] = sma[i] + np.std(input[i-period:i])

        return sma
    
    def std_dev(self, input, period):
        p_2sig = np.zeros_like(input)
        p_2sig[0:period] = np.sum(input[0:period]) / len(input[0:period])

        m_2sig = np.zeros_like(input)
        m_2sig[0:period] = np.sum(input[0:period]) / len(input[0:period]) 

        for i in np.arange(period, len(input), 1):
            avg = np.sum(input[i-period:i]) / period
            p_2sig[i] = avg + 2 * np.std(input[i-period:i])
            m_2sig[i] = avg - 2 * np.std(input[i-period:i])

        return p_2sig, m_2sig
    
    
    def vwap_boot(self, input, period, sample_size=2000, stddev=False):
        self.vwap_boot_arr = np.zeros_like(input)
        vwap_plus_sigma = np.zeros_like(input)
        vwap_minus_sigma = np.zeros_like(input)

       #Initialize starting values in first period
        self.vwap_boot_arr[0:period] = np.sum(self.volume[0:period] * input[0:period]) / np.sum(self.volume[0:period])
        vwap_plus_sigma[0:period] = self.vwap_boot_arr[0:period]
        vwap_minus_sigma[0:period] = self.vwap_boot_arr[0:period]

        for i in np.arange(period, len(input), 1):
            volume_init = self.volume[i - period:i]
            price_init = input[i - period:i]

            #Apply bootstrapping to data range
            indices = np.random.choice(range(len(volume_init)), size=sample_size, replace=True)
            volume_resample = volume_init[indices]
            price_resample = price_init[indices]

            self.vwap_boot_arr[i] = np.sum(volume_resample * price_resample) / np.sum(volume_resample)

            if stddev == True:
                stddev_vol = self.stddev(volume_resample)
                vol_plus_sigma = volume_resample + (2 * stddev_vol)
                vol_minus_sigma = volume_resample - (2 * stddev_vol)

                stddev_price = self.stddev(price_resample)
                price_plus_sigma = price_resample + (2 * stddev_price)
                price_minus_sigma = price_resample - (2 * stddev_price)

                vwap_plus_sigma[i] = np.sum(vol_plus_sigma * price_plus_sigma) / np.sum(vol_plus_sigma)
                vwap_minus_sigma[i] = np.sum(vol_minus_sigma * price_minus_sigma) / np.sum(vol_minus_sigma)

        if stddev == True:
            return self.vwap_boot_arr, vwap_plus_sigma, vwap_minus_sigma
        
        else:
            return self.vwap_boot_arr

    def E_sum(self, x, f):
        if len(x) != len(f):
            raise ValueError('Bins and frequency arrays must be the same length.')
        
        else:
            E_val = 0

            for i, j in zip(x, f):
                E_val += i * j


            return(E_val)

    def stddev(self, x):
        probs, vals = np.histogram(x)
        probs = probs / len(x)
        vals_square = vals **2

        sig_square = self.E_sum(vals_square[:-1], probs) - (self.E_sum(vals[:-1], probs)) **2 

        return np.sqrt(sig_square)


        