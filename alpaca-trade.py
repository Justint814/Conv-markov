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

emd = EMD()


API_KEY = "AKBRO077E0RQ6IQYAL72"
API_SECRET = "0TvY1exvXGXfsruRR8GUGQduDJzpEGAZqFWFRasL"
SYMBOL = 'MSFT'

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

def sma(input, period, sample_size=200):
        sma = np.zeros_like(input)

        #Initialize starting values in first period
        sma[0:period] = np.sum(input[0:period]) / len(input[0:period])

        for i in np.arange(period, len(input), 1):
            sma[i] = np.sum(input[i-period:i]) / period

        return sma

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

#Function to predict next average price over a certain range by satisfying the condition that the average of the 2nd derivative is equal to zero
def DMA(data, period, step=1):
    dma_arr = np.zeros_like(data)

    init_arr = data[0:period:step]
    dma_arr[0:period] = init_arr[-1] + init_arr[-1] - init_arr[0]

    for i in np.arange(period, len(data), 1):
        start = i - period
        it_arr = data[start:i:step]

        dma_sum = it_arr[-1] + it_arr[1] - it_arr[0]
        dma_arr[i] = dma_sum

    return(dma_arr)

def DMA_hist(data, period, step=1): #  Returns arrays of data-dma_data, mean at each point, with size N, +-1sigma and +-2sigma.
    N = 1000
    dma_data = DMA(data, period, step=step)
    diff_arr = data - dma_data

    if len(data) > N:
        mean = np.zeros_like(data)
        plus_sig = np.zeros_like(data)
        minus_sig = np.zeros_like(data)
        plus_2sig = np.zeros_like(data)
        minus_2sig = np.zeros_like(data)

        #Initialize values smaller than N, so they show up in roughly the same area when plotted
        mean[0:N] = np.mean(diff_arr[0:N])
        plus_sig[0:N] = mean[0:N]
        minus_sig[0:N] = mean[0:N]
        plus_2sig[0:N] = mean[0:N]

        for i in np.arange(N, len(diff_arr), 1):
            start = i - N
            std_dev = np.std(diff_arr[start:i])

            mean[i] = np.mean(diff_arr[start:i])
            plus_sig[i] = mean[i] + std_dev
            plus_2sig[i] = mean[i] + 2 * std_dev
            minus_sig[i] = mean[i] - std_dev
            minus_2sig[i] = mean[i] - 2 * std_dev

        return diff_arr, mean, plus_sig, minus_sig, plus_2sig, minus_2sig

    else:
        raise ValueError(f"Input arrays must be of size {N} or greater")

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
        new_volume = np.array([bar.volume], dtype=np.float64)

        historical_prices = np.append(historical_prices, new_price)
        historical_volume = np.append(historical_volume, new_volume)
        length = range(len(historical_prices))

        print("length of data:", len(historical_prices))

        if historical_prices.shape[0] > 2000:
            historical_prices = historical_prices[-2000:]
            historical_volume = historical_volume[-2000:]

        #Calculate neccessary values
        #EMD DATA:
        '''
        count = 1
        for i in np.array([100]):
            vwap_combo = historical_prices

            for j in np.arange(50,i,1):
                vwap_combo = vwap_combo + vwap(historical_prices, historical_volume, j)
                count += 1

            #IMFs = emd.emd(vwap_combo)
            #imf_ema = ema(IMFs[3], 7)
            '''
        ema_data = ema(historical_prices, 12)
        diff_arr, mean, plus_sig, minus_sig, plus2_sig, minus2_sig = DMA_hist(ema_data, 25) #20


        
        #IMF_long = emd.emd(ema(historical_prices, 12))

        #BOOTED VWAP DATA:
        vwap_20_boot = vwap(ema_data, historical_volume, 20)

        length = np.arange(len(historical_prices))

        if plot == True:
            '''
            fig = go.Figure() #Make the figure
            fig.add_trace(go.Scatter(x=length, y=IMFs[4] + IMFs[5], mode='lines', name=f"IMF4 {SYMBOL}")) #Add data to the figure

            fig.add_trace(go.Scatter(x=length, y=ema(IMFs[4] + IMFs[5], 9), mode='lines', name=f"EMA4 {SYMBOL}"))

            fig.update_traces(line=dict(width=.6)) #Update line width

            fig.update_layout(
                title=f"{SYMBOL} 4",
                xaxis_title="time (min)",
                yaxis_title="Amplitude"
            ) #Add titles

            fig.show() #Show the figure
            '''

            fig = go.Figure() #Make the figure

            fig.add_trace(go.Scatter(x=length, y=diff_arr, mode='lines', name="Price-DMA")) #Add data to the figure ##2 15s
            fig.add_trace(go.Scatter(x=length, y=mean, mode='lines', name="mean"))
            fig.add_trace(go.Scatter(x=length, y=plus_sig, mode='lines', name="+ \u03C3"))
            fig.add_trace(go.Scatter(x=length, y=minus_sig, mode='lines', name="- \u03C3"))
            fig.add_trace(go.Scatter(x=length, y=plus2_sig, mode='lines', name="+ 2 \u03C3"))
            fig.add_trace(go.Scatter(x=length, y=minus2_sig, mode='lines', name="- 2 \u03C3"))
            fig.update_traces(line=dict(width=.6)) #Update line width

            fig.update_layout(
                title="DMA Hist",
                xaxis_title="tick",
                yaxis_title="Amplitude"
            ) #Add titles

            fig.show() #Show the figure


            
            fig = go.Figure() #Make the figure
            fig.add_trace(go.Scatter(x=length, y=historical_prices, mode='lines')) #Add data to the figure
            #fig.add_trace(go.Scatter(x=length, y=vwap_20_boot, mode='lines'))
            #fig.add_trace(go.Scatter(x=length, y=vwap_80_boot, mode='lines'))
            #fig.add_trace(go.Scatter(x=length, y=ema(historical_prices, 25), mode='lines'))
            #fig.add_trace(go.Scatter(x=length, y=sma(historical_prices, 50), mode='lines')) 
            fig.add_trace(go.Scatter(x=length, y=DMA(ema_data, 25), mode='lines'))
            fig.add_trace(go.Scatter(x=length, y=ema(historical_prices, 12), mode='lines'))

            fig.update_traces(line=dict(width=.6)) #Update line width

            fig.update_layout(
                title=f"Chart {SYMBOL}",
                xaxis_title="time (min)",
                yaxis_title="Amplitude"
            ) #Add titles

            fig.show()

            bar_queue.task_done()

async def on_bar(bar):
    # Just enqueue the new bar immediately
    bar_queue.put(bar)

def main():
    global historical_prices, historical_volume
    
    # 1. Fetch historical data
    hist_client = StockHistoricalDataClient(API_KEY, API_SECRET)
    end_time = datetime.now(pytz.UTC)
    start_time = end_time - timedelta(minutes=10000)
    print(start_time)

    req = StockBarsRequest(
        symbol_or_symbols=SYMBOL,
        timeframe=TimeFrame.Minute,
        start=start_time,
        end=end_time,
        limit=10000,
        feed="iex"
    )

    bars = hist_client.get_stock_bars(req).df
    historical_prices = bars['close']
    historical_volume = bars['volume']
    '''
    historical_prices = np.array([bar.close for bar in bars], dtype=np.float64)
    historical_volume = np.array([bar.volume for bar in bars], dtype=np.float64)
    '''
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
