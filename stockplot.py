import numpy as np
from stockstream import PolygonData
import matplotlib.pyplot as plt
from PyEMD import EMD
import plotly.graph_objects as go
from scipy.signal import hilbert

emd = EMD()

API_key = 'aSE_Q2Vmnh9BCPSumGUceCJphILlB3ag'
ticker = 'SPY'
limit = 30000
save_dir = "./data/"
price_name = save_dir + ticker + "-min.npy"
volume_name = save_dir + ticker + "-min-volume.npy"

stream_obj = PolygonData(API_key)

min_data = stream_obj.retrieve_data(ticker, limit)
volume_data = stream_obj.volume


#min_data = np.load(price_name)[3000:3800:1]
#volume_data = np.load(volume_name)[3000:3800:1]



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

    if len(data) > 999:
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
        raise ValueError("Input arrays must be of size 1000 or greater")
    
def diff_stats(prices, ind_prices):
    N = 1000
    diff_arr = prices - ind_prices

     #Initialize values smaller than N, so they show up in roughly the same area when plotted
    mean = np.zeros_like(prices)
    plus_sig = np.zeros_like(prices)
    minus_sig = np.zeros_like(prices)
    plus_2sig = np.zeros_like(prices)
    minus_2sig = np.zeros_like(prices)
    plus_3sig = np.zeros_like(prices)
    minus_3sig = np.zeros_like(prices) 

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


    
ema_data = stream_obj.boot_sma(min_data, 100)
diff_arr2, mean2, plus_sig2, minus_sig2, plus_2sig2, minus_2sig2 = diff_stats(min_data, ema_data)
dma_dat = DMA(ema_data, 100)
diff_arr, mean, plus_sig, minus_sig, plus2_sig, minus2_sig = DMA_hist(ema_data, 100)


stream_obj.volume = volume_data
stream_obj.output = min_data

#vwap_data = stream_obj.vwap(min_data, 200)
vwap_boot = stream_obj.vwap_boot(ema_data, 10, sample_size=20000)
vwap_boot2= stream_obj.vwap_boot(vwap_boot, 25, sample_size=20000)


xdata = range(len(min_data))

IMFs = []
for i in np.array([100]):
    vwap_combo = min_data

    for j in np.arange(1,i,1):
        vwap_combo = vwap_combo + stream_obj.ema(min_data, j)
    
    IMFs.append(emd.emd(vwap_combo))

IMFs = IMFs[0]
IMF_combo = IMFs[4] + IMFs[5]

IMF_long = emd.emd(stream_obj.ema(min_data, 12))


#vwap 1-100, only trade towards vwap with 20 minute opposite of direction you are trading compared to 80
#vwap 1-20, faster changes

#Plot pts with plotly interactive plot
length = np.arange(len(xdata))

fig = go.Figure() #Make the figure
fig.add_trace(go.Scatter(x=length, y=diff_arr, mode='lines', name="Price-DMA")) #Add data to the figure ##2 15s
fig.add_trace(go.Scatter(x=length, y=mean, mode='lines', name="mean"))
fig.add_trace(go.Scatter(x=length, y=plus_sig, mode='lines', name="+ \u03C3"))
fig.add_trace(go.Scatter(x=length, y=minus_sig, mode='lines', name="- \u03C3"))
fig.add_trace(go.Scatter(x=length, y=plus2_sig, mode='lines', name="+ 2 \u03C3"))
fig.add_trace(go.Scatter(x=length, y=minus2_sig, mode='lines', name="- 2 \u03C3"))
fig.add_trace(go.Scatter(x=length, y=stream_obj.ema(diff_arr, 20)))
fig.add_trace(go.Scatter(x=length, y=stream_obj.ema(diff_arr, 25)))

fig.update_traces(line=dict(width=.6)) #Update line width

fig.update_layout(
    title="DMA Hist",
    xaxis_title="tick",
    yaxis_title="Amplitude"
) #Add titles

fig.show() #Show the figure

fig = go.Figure() #Make the figure
fig.add_trace(go.Scatter(x=length, y=diff_arr2, mode='lines', name="Price-DMA")) #Add data to the figure ##2 15s
fig.add_trace(go.Scatter(x=length, y=mean2, mode='lines', name="mean"))
fig.add_trace(go.Scatter(x=length, y=plus_sig2, mode='lines', name="+ \u03C3"))
fig.add_trace(go.Scatter(x=length, y=minus_sig2, mode='lines', name="- \u03C3"))
fig.add_trace(go.Scatter(x=length, y=plus_2sig2, mode='lines', name="+ 2 \u03C3"))
fig.add_trace(go.Scatter(x=length, y=minus_2sig2, mode='lines', name="- 2 \u03C3"))
fig.add_trace(go.Scatter(x=length, y=stream_obj.ema(diff_arr2, 19)))
fig.add_trace(go.Scatter(x=length, y=stream_obj.ema(diff_arr2, 25)))

fig.update_traces(line=dict(width=.6)) #Update line width

fig.update_layout(
    title="SMA Hist",
    xaxis_title="tick",
    yaxis_title="Amplitude"
) #Add titles

fig.show() #Show the figure


fig = go.Figure() #Make the figure
fig.add_trace(go.Scatter(x=length, y=min_data, mode='lines')) #Add data to the figure
#fig.add_trace(go.Scatter(x=length, y=vwap_boot, mode='lines'))
#fig.add_trace(go.Scatter(x=length, y=vwap_boot2, mode='lines'))
fig.add_trace(go.Scatter(x=length, y=ema_data, mode='lines'))
fig.add_trace(go.Scatter(x=length, y=dma_dat, mode='lines'))
fig.add_trace(go.Scatter(x=length, y=stream_obj.std_dev(min_data, 150)[0], mode='lines'))
fig.add_trace(go.Scatter(x=length, y=stream_obj.std_dev(min_data, 150)[1], mode='lines'))
fig.add_trace(go.Scatter(x=length, y=stream_obj.ema(min_data, 9), mode='lines'))
fig.add_trace(go.Scatter(x=length, y=stream_obj.ema(min_data, 12), mode='lines'))


fig.update_traces(line=dict(width=.6)) #Update line width

fig.update_layout(
    title=f"Chart {ticker}",
    xaxis_title="time (min)",
    yaxis_title="Amplitude"
) #Add titles

fig.show()

'''
fig = go.Figure() #Make the figure
fig.add_trace(go.Scatter(x=length, y=IMF_long[4], mode='lines')) #Add data to the figure
fig.add_trace(go.Scatter(x=length, y=stream_obj.ema(IMF_long[4], 7), mode='lines')) #Add data to the figure

fig.update_traces(line=dict(width=.6)) #Update line width

fig.update_layout(
    title=f"Chart {ticker}",
    xaxis_title="time (min)",
    yaxis_title="Amplitude"
) #Add titles

fig.show()
'''

'''
for i in range(np.shape(IMFs)[0]):
    analytic_signal = hilbert(IMFs[i,:])
    sampling_rate = 1
    instantaneous_phase = np.unwrap(np.angle(analytic_signal))
    instantaneous_frequency = (np.diff(instantaneous_phase) / (2.0 * np.pi)) * sampling_rate

    time = np.arange(len(min_data))
    #inst_freq = np.array([time, instantaneous_frequency])

    fig = go.Figure() #Make the figure
    fig.add_trace(go.Scatter(x=time, y=instantaneous_frequency, mode='lines', name=i)) #Add data to the figure

    fig.update_traces(line=dict(width=.6)) #Update line width

    fig.update_layout(
        title=i,
        xaxis_title="time (min)",
        yaxis_title="Frequency"
    ) #Add titles 

    fig.show()
'''
#if most recent vwap extrema from last transition of vwaps is a certain percentage of price, enter at vwap crossings and sell at vwap extrema
