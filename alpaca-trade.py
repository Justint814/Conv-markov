import threading
import queue
import numpy as np
from datetime import datetime, timedelta
import pytz
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.live import StockDataStream
import time
from PyEMD import EMD
import plotly.graph_objects as go
import matplotlib.pylab as plt

API_KEY = "AK4C2M4DAF4XDLPVWNXJ"
API_SECRET = "CTh1PNJZztBZtZysMBnHzd0yqAtWBg3VijBMTncK"
SYMBOL = "AAPL"

def vwap_boot(input, volume, period, sample_size=2000, stddev=False):
    vwap_boot_arr = np.zeros_like(input)

    #Initialize starting values in first period
    vwap_boot_arr[0:period] = np.sum(volume[0:period] * input[0:period]) / np.sum(volume[0:period])

    for i in np.arange(period, len(input), 1):
        volume_init = volume[i - period:i]
        price_init = input[i - period:i]

        #Apply bootstrapping to data range
        indices = np.random.choice(range(len(volume_init)), size=sample_size, replace=True)
        volume_resample = volume_init[indices]
        price_resample = price_init[indices]

        vwap_boot_arr[i] = np.sum(volume_resample * price_resample) / np.sum(volume_resample)
        
    return vwap_boot_arr

def vwap(input, volume, period):
    vwap_arr = np.zeros_like(input)

    #Initialize starting values in first period
    vwap_arr[0:period] = np.sum(volume[0:period] * input[0:period]) / np.sum(volume[0:period])

    for i in np.arange(period, len(input), 1):
        vwap_arr[i] = np.sum(volume[i - period:i] * input[i - period:i]) / np.sum(volume[i - period:i])

    return vwap_arr
        
def ema(input, period):
        ema_arr = np.zeros_like(input)
        multiplier = 2 / (period + 1)

        #Initialize first EMA value as SMA
        ema_val = np.sum(input[0:period]) / period
        ema_arr[0:period] = [ema_val for i in range(period)]

        for i in np.arange(period, len(input), 1):
            ema_val = (input[i] * multiplier) + ema_val * (1 - multiplier)
            ema_arr[i] = ema_val

        return ema_arr

# Shared queue to pass bars from WebSocket to processing thread
bar_queue = queue.Queue()

# Global NumPy array for price data
historical_prices = np.array([], dtype=np.float64)
historical_volume = np.array([], dtype=np.float64)

def data_processor():
    plot = True
    global historical_prices, historical_volume
    while True:
        bar = bar_queue.get()  # Wait for next bar
        if bar is None:
            break  # Exit signal
        
        # Append new close price
        new_price = np.array([bar.close], dtype=np.float64)
        new_volume = np.array([bar.volume], dtpye=np.float64)

        historical_prices = np.append(historical_prices, new_price)
        historical_volume = np.append(historical_volume, new_volume)
        length = range(len(historical_prices))

        if historical_prices.shape[0] > 500:
            historical_prices = historical_prices[-500:]
            historical_volume = historical_volume[-500:]

        #Calculate neccessary values
        #EMD DATA:
        imf_3 = []
        imf_ema = []
        for i in np.array([10, 15, 20, 100]):
            vwap_combo = historical_prices

            for j in np.arange(1,i,1):
                vwap_combo = vwap_combo + vwap(historical_prices, historical_volume, j)
            
            emd = EMD()
            IMFs = emd.emd(vwap_combo)
            imf_ema.append(ema(IMFs[3], 8))
            imf_3.append(IMFs[3])

        imf_3 = np.array(imf_3)
        imf_ema = np.array(imf_ema)

        #BOOTED VWAP DATA:
        vwap_20_boot = vwap_boot(historical_prices, historical_volume, period=20)
        ema_data = ema(historical_prices, 14)
        vwap_20_boot = vwap_boot(ema_data, 20, historical_volume, sample_size=20000)
        vwap_80_boot = vwap_boot(vwap_20_boot, 80, historical_volume, sample_size=20000)

        #Trade Conditions:
        vwap_diff = vwap_80_boot[-1] - vwap_20_boot[-1]
        price_vwap_diff = vwap_20_boot[-1] - historical_prices[-1]
        imf_diff = imf_3[-1] - imf_ema[-1]
        imf_prev_diff = imf_ema[-2] - imf_3[-2]

        if vwap_diff and price_vwap_diff and imf_diff and imf_prev_diff > 0:
            print(f"Up signal on {SYMBOL} at {time.time}")

        elif vwap_diff and price_vwap_diff and imf_diff and imf_prev_diff < 0: 
            print(f"Down signal on {SYMBOL} at {time.time}")
        
        else:
            pass

        if plot == True:
            for i in range(4):
                fig = go.Figure() #Make the figure
                fig.add_trace(go.Scatter(x=length, y=imf_3[i], mode='lines', name=i)) #Add data to the figure

                fig.add_trace(go.Scatter(x=length, y=imf_ema[i], mode='lines'))

                fig.update_traces(line=dict(width=.6)) #Update line width

                fig.update_layout(
                    title=i,
                    xaxis_title="time (min)",
                    yaxis_title="Amplitude"
                ) #Add titles

                fig.show() #Show the figure

            plt.plot(length, historical_prices, label='1min Close Price')
            plt.plot(length, vwap_20_boot, label='18min Bootstrapped VWAP', color="red")
            plt.plot(length, vwap_80_boot)

            plt.show()


        bar_queue.task_done()

def on_bar(bar):
    # Just enqueue the new bar immediately
    bar_queue.put(bar)

def main():
    global historical_prices, historical_volume
    
    # 1. Fetch historical data
    hist_client = StockHistoricalDataClient(API_KEY, API_SECRET)
    end_time = datetime.now(pytz.UTC)
    start_time = end_time - timedelta(minutes=500)

    req = StockBarsRequest(
        symbol_or_symbols=SYMBOL,
        timeframe=TimeFrame.Minute,
        start=start_time,
        end=end_time,
        limit=500
    )

    bars = hist_client.get_stock_bars(req)
    historical_prices = np.array([bar.close for bar in bars[SYMBOL]], dtype=np.float64)
    historical_volume = np.array([bar.volume for bar in bars[SYMBOL]], dtype=np.float64)
    print(f"Loaded {historical_prices.shape[0]} historical closes.")

    # 2. Start processing thread
    processor_thread = threading.Thread(target=data_processor, daemon=True)
    processor_thread.start()

    # 3. Start live stream
    stream = StockDataStream(API_KEY, API_SECRET, raw_data=False)
    stream.subscribe_bars(on_bar, SYMBOL)
    print("Streaming live 1-minute bars...")
    stream.run()

    # 4. Cleanup (never reached here normally)
    bar_queue.put(None)
    processor_thread.join()

if __name__ == "__main__":
    main()
