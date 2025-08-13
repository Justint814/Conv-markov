import numpy as np
from stockstream import PolygonData
import matplotlib.pyplot as plt
from PyEMD import EMD
import plotly.graph_objects as go
from scipy.signal import hilbert
API_key = 'aSE_Q2Vmnh9BCPSumGUceCJphILlB3ag'
ticker = 'SPY'
limit = 10000
save_dir = "./data/"
price_name = save_dir + ticker + "-min.npy"
volume_name = save_dir + ticker + "-min-volume.npy"

stream_obj = PolygonData(API_key)

min_data = stream_obj.retrieve_data(ticker, limit)[2200:2700]
volume_data = stream_obj.volume[2200:2700]


#min_data = np.load(price_name)[0:2000:1]
#volume_data = np.load(volume_name)[0:2000:1]

stream_obj.volume = volume_data
stream_obj.output = min_data

#vwap_data = stream_obj.vwap(min_data, 200)
ema_data = stream_obj.ema(min_data, 14)
vwap_boot = stream_obj.vwap_boot(ema_data, 10, sample_size=20000)
vwap_boot2= stream_obj.vwap_boot(vwap_boot, 25, sample_size=20000)


#ema2 = stream_obj.ema(min_data, 12)
#ema3 = stream_obj.ema(min_data, 9)


xdata = range(len(min_data))

imf_3 = []
imf_ema_3 = []
imf_4 = []
imf_ema_4 = []
for i in np.array([5, 10, 15, 20, 100]):
    vwap_combo = min_data

    for j in np.arange(1,i,1):
        vwap_combo = vwap_combo + stream_obj.vwap(min_data, j)
    

    emd = EMD()
    IMFs = emd.emd(vwap_combo)

    imf_ema_3.append(stream_obj.ema(IMFs[3], 7))
    imf_3.append(IMFs[3])

    imf_ema_4.append(stream_obj.ema(IMFs[4], 7))
    imf_4.append(IMFs[4])

np.array(imf_3)
np.array(imf_ema_3)
np.array(imf_4)
np.array(imf_ema_4)

#vwap 1-100, only trade towards vwap with 20 minute opposite of direction you are trading compared to 80
#vwap 1-20, faster changes

#Plot pts with plotly interactive plot
length = np.arange(len(xdata))

fig = go.Figure() #Make the figure
fig.add_trace(go.Scatter(x=length, y=imf_3[2], mode='lines', name="imf3")) #Add data to the figure

fig.add_trace(go.Scatter(x=length, y=imf_ema_3[2], mode='lines'))

fig.update_traces(line=dict(width=.6)) #Update line width

fig.update_layout(
    title="imf3",
    xaxis_title="time (min)",
    yaxis_title="Amplitude"
) #Add titles

fig.show() #Show the figure


fig = go.Figure() #Make the figure
fig.add_trace(go.Scatter(x=length, y=imf_4[0], mode='lines', name="imf4")) #Add data to the figure

fig.add_trace(go.Scatter(x=length, y=imf_ema_4[0], mode='lines'))

fig.update_traces(line=dict(width=.6)) #Update line width

fig.update_layout(
    title="imf4",
    xaxis_title="time (min)",
    yaxis_title="Amplitude(4)"
) #Add titles

fig.show() #Show the figure

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


plt.plot(xdata, min_data, label='1min Close Price')
plt.plot(xdata, vwap_boot, label='18min Bootstrapped VWAP', color="red")
plt.plot(xdata, vwap_boot2)

plt.show()

#if most recent vwap extrema from last transition of vwaps is a certain percentage of price, enter at vwap crossings and sell at vwap extrema
