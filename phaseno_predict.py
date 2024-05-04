# %% [markdown]
# ###
# ## Phase Neural Operator (PhaseNO)
# ### PhaseNO for multi-station earthquake detection and phase picking
# ### - Predict on continuous seismic data with the pretrained model - 
# 
# ##### Author: Hongyu Sun
# ##### Email: hongyu-sun@outlook.com

# %%
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import gc 
import numpy as np
import h5py
import torch
import pandas as pd
import json
import random; random.seed(10)
import obspy
from tqdm import tqdm
from datetime import datetime, timedelta
from collections import namedtuple
from scipy.signal import butter, filtfilt

import matplotlib.pyplot as plt
import matplotlib
font = {'weight' : 'bold', 
        'size'   : 12}
matplotlib.rc('font', **font)

from models import PhaseNO
from utils import _trim_nan, _detect_peaks, extract_amplitude
from utils import slide_window, save_picks, save_picks_json

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %% [markdown]
# ### Set data path and parameters here

# %%
Data_Path   = '/nfs/quakeflow_dataset/NC/waveform_h5/2020.h5' # folder contains raw data
station_info = './test_nc2020/nc2020_test_stations.csv'

if not os.path.exists(station_info):
    # Create an empty dataframe
    df = []
    station_ids = []

    with h5py.File(Data_Path, 'r') as fp:
        for key in tqdm(fp.keys()):
            # Get the group corresponding to the key
            event = fp[key]

            # Iterate over all keys in the group
            for station in event.keys():
                if station in station_ids:
                    continue
                # Get the attributes of the station
                attrs = event[station].attrs
                info = {}
                for k in attrs.keys():
                    if k in ['component', 'depth_km', 'dt_s', 'elevation_m', 'instrument', 'latitude', 'local_depth_m', 'location', 'longitude', 'network', 'station', 'unit']:
                        info[k] = attrs[k]

                # Convert the attributes to a pandas dataframe
                attrs_df = pd.DataFrame(info, index=[station])

                # Append the dataframe to the main dataframe
                df.append(attrs_df)
                station_ids.append(station)

    # Concatenate all dataframes in the list
    df = pd.concat(df)
    # Drop duplicate rows based on the index (key)
    df = df[~df.index.duplicated(keep='first')]
    df = df.reset_index()

    # Rename the columns
    df.columns = ['station_id'] + list(df.columns[1:])
    df = df[['station_id', 'network', 'station', 'instrument', 'latitude', 'longitude', 'depth_km', 'elevation_m','local_depth_m', 'location', 'component', 'dt_s', 'unit']]
    df.to_csv(station_info, index=False)

PROB = 0 # use 1 if you want to output predicted probability time series and then plotting with phaseno_plot.ipynb
# consider PROB = 0 to save storage if processing a large amount of data 
PICK = 1 # use 1 to output picks using a pikcing threshold (default threshold is 0.3)

if PROB == 1:
    output_name = './test_nc2020/results/probability'
    prob_dir = os.path.join(output_name)
    if not os.path.exists(prob_dir):
        os.makedirs(prob_dir)

if PICK == 1:
    result_name = './test_nc2020/results/picks'
    result_dir = os.path.join(result_name)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

threshold = 0.3 # probability threshold for picking
num_station_split = 20 # number of stations in a run 
num_station_onerun_max = 20 # maximum number of stations in a run on a GPU
in_samples = 3000 # time window is 30 s in a run
overlap = 1000 # use a 10-s overlap between two time windows
steps = in_samples - overlap
sample_rate = 100
highpass_filter = 1 # Hz
eps = 1e-6
dtype = 'float32'
n_channel = 3

# %% [markdown]
# ### set station information: path and center
# #### Important! modify "center" to the correct center of your seismic networks

# %%
stations_new = pd.read_csv(station_info, sep=',')
station_location = stations_new[['longitude','latitude']].values
set_center = lambda station_location: (round(min(station_location[:,0].min()+1, np.mean(station_location[:,0])), 2), round(min(station_location[:,1].min()+1, np.mean(station_location[:,1])), 2))
center = set_center(station_location)
x_min, y_min = center[0] - 1, center[1] - 1
stations_new.set_index('station_id', inplace=True)

PLOT = 0

def plot_station(stations):
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    for i,sta in enumerate(stations['station']):
        ax.plot(stations['longitude'][i],stations['latitude'][i],'^',markersize=10)
        ax.text(stations['longitude'][i],stations['latitude'][i],sta)

    ax.locator_params(axis='both', nbins=6)
    ax.set_xlabel('longitude')
    ax.set_ylabel('latitude')
    # ax.set_xlim(x_min, x_min+2); 
    # ax.set_ylim(y_min, y_min+2)

if PLOT == 1: 
    plot_station(stations_new)

# %% [markdown]
# ### load trained PhaseNO model

# %%
model = PhaseNO.load_from_checkpoint(checkpoint_path='./models/epoch=19-step=1140000.ckpt', map_location=device).to(device)
model.eval()

# %% [markdown]
# ### start predicting and picking

# %%
label_name=["P_Prob","S_Prob"]
record = namedtuple("phase", ["fname", "station_id", "t0", "p_idx", "p_prob", "s_idx", "s_prob"])
nc2020 = h5py.File(Data_Path, 'r')

# highpass filter
b, a = butter(4, highpass_filter/(sample_rate/2), 'highpass')

# loop over hourly data segment
for event_id in list(nc2020.keys()):
    event = nc2020[event_id]
    stations = list(event.keys())
    manual_stations = [sta for sta in stations if event[sta].attrs['phase_status'] == 'manual']
    auto_stations = [sta for sta in stations if sta not in manual_stations]
    picks_all_stations = []
    amps_all_stations  = []

    print('current at', event_id)

    station_all = list(event.keys())

    print('number of stations: ', len(station_all))

    while len(station_all) > 0:

        if len(station_all) <= num_station_onerun_max:
            station_select = station_all[:len(station_all)]
            for x in station_select:
                station_all.remove(x)
    
        else:
            station_select = manual_stations + random.sample(auto_stations, num_station_split-len(manual_stations))
            station_all = []
    
        print('selected station in one sample: ', station_select)
    
        df = stations_new.loc[station_select]
        station_location = df[['longitude','latitude']].values
        center = set_center(station_location)
        x_min, y_min = center[0] - 1, center[1] - 1

        num_station = len(station_select)

        station_convert = station_location
        station_convert = (station_location - [x_min, y_min]) / 2
        station_convert = torch.from_numpy(station_convert).float()

        # generate edge_index
        row_a=[]
        row_b=[]  
        row_ix=[]
        row_iy=[]
        row_jx=[]
        row_jy=[]
        for i in range(num_station):
            for j in range(num_station):
                row_a.append(i)
                row_b.append(j)
                row_ix.append(station_convert[i,0])
                row_iy.append(station_convert[i,1])
                row_jx.append(station_convert[j,0])
                row_jy.append(station_convert[j,1])

        edge_index=[row_a,row_b,row_ix,row_iy,row_jx,row_jy]
        edge_index = torch.from_numpy(np.array(edge_index)).float().to(device)
        
        picks_select_stations = []
        print('------------------------ reading waveforms ------------------------')
        nt = event[station_select[0]][:].shape[1]
        waveforms = np.zeros((num_station, 3, nt))
        for i, sta_id in enumerate(station_select):
            sta = event[sta_id]
            waveforms[i] = sta[:,:nt]

        starttime = datetime.strptime(event.attrs['begin_time'], "%Y-%m-%dT%H:%M:%S.%f")
        endtime = datetime.strptime(event.attrs['end_time'], "%Y-%m-%dT%H:%M:%S.%f")

        ##### process raw amplitude for magnitude estimation #####  

        data_raw = np.zeros((num_station, 3, nt))
        for i, sta_id in enumerate(station_select):
            sta = event[sta_id]
            tmp = sta[:,:nt]
            if sta.attrs["unit"][-6:] == "m/s**2":
                tmp = np.cumsum(tmp*sta.attrs['dt_s'], axis=1)
                # highpass filter
                if highpass_filter > 0:
                    tmp = filtfilt(b, a, tmp, axis=1)
            data_raw[i] = tmp

        preds = []
        starts = []

        t0 = starttime.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]
        print('t0=', t0)

        print('------------------------ start predicting ------------------------')

        for (windowed_st,idx) in slide_window(waveforms, in_samples, steps, axis=2): 
            start_trim = starttime + timedelta(seconds=idx/sample_rate)
            end_trim = start_trim + timedelta(seconds=in_samples/sample_rate)
            temp = windowed_st.astype(float)

            temp_mean = np.mean(temp,axis=-1,keepdims=True)
            temp_std  = np.std(temp,axis=-1,keepdims=True)
            temp_std[temp_std==0] = 1
            temp = (temp-temp_mean)/(temp_std+eps)

            X = np.zeros((num_station, 3+2, in_samples))
            X[:,:3,:] = temp/10

            for istation in np.arange(num_station):
                X[istation,3,:] = station_convert[istation,0]
                X[istation,4,:] = station_convert[istation,1]

            X = torch.tensor(X, dtype=torch.float)
            X = X.to(device)
            res=torch.sigmoid(model.forward((X,None,edge_index)))
            preds.append(res.cpu().detach().numpy())
            starts.append(idx)

        print('------------------------ start post-processing ------------------------')

        gc.collect()

        prediction_sample_factor=1
        # Maximum number of predictions covering a point
        coverage = int(
            np.ceil(in_samples / (in_samples - overlap) + 1)
        )
        
        pred_length = int(
            np.ceil(
                (np.max(starts)+in_samples) * prediction_sample_factor
            )
        )
        pred_merge = (
            np.zeros_like(
                preds[0], shape=( preds[0].shape[0],preds[0].shape[1], pred_length, coverage)
            )
            * np.nan
        )

        for i, (pred, start) in enumerate(zip(preds, starts)):
            pred_start = int(start * prediction_sample_factor)
            pred_merge[
                :,:, pred_start : pred_start + pred.shape[2], i % coverage
            ] = pred

        del preds
        gc.collect()

        if PICK == 1:

            p_idx, p_prob, s_idx, s_prob = [], [], [], []
            picks_part_stations = []
            for k in range(num_station):
                id = station_select[k]
                sta = station_select[k]
                pred = np.nanmean(pred_merge[k], axis=-1)
                P_seq, _, _ = _trim_nan(pred[0])
                S_seq, _, _ = _trim_nan(pred[1])
                p, p_pro =_detect_peaks(P_seq[:12000], mph=threshold, mpd=1.0*sample_rate)
                s, s_pro =_detect_peaks(S_seq[:12000], mph=threshold, mpd=1.0*sample_rate)

                picks_part_stations.append(record(id, sta, t0, list(p), list(p_pro), list(s), list(s_pro)))

            amps_part_stations = extract_amplitude(data_raw, picks_part_stations, window_p=8, window_s=4, dt=1/sample_rate)
            
            picks_all_stations += picks_part_stations
            amps_all_stations += amps_part_stations

        if PROB == 1:
            for k in range(num_station):

                #id = fname[k]
                output = obspy.Stream()
                pred = np.nanmean(pred_merge[k], axis=-1)

                for i in range(2):

                    trimmed_pred, f, _ = _trim_nan(pred[i])
                    trimmed_start = starttime + f / sample_rate

                    output.append(
                        obspy.Trace(
                            trimmed_pred,
                            {
                                "starttime": trimmed_start,
                                "sampling_rate": sample_rate,
                                "network": df.loc[station_select[k]]['network'],
                                "station": df.loc[station_select[k]]['station'],
                                "location": df.loc[station_select[k]]['location'],
                                "channel": label_name[i],
                            },
                        )
                    )

                output.write(os.path.join(prob_dir, station_select[k]+'.mseed'), format='MSEED')
                del output
                gc.collect()

        del pred_merge
        gc.collect()

        print('------------------------select stations done ------------------------')

    print('========================',event_id,' done ========================')

    if PICK == 1:
        save_picks(picks_all_stations, result_dir, dt=1/sample_rate, amps=amps_all_stations,fname=event_id+'.csv')
        #save_picks_json(picks_all_stations, result_dir, dt=1/sample_rate, amps=amps_all_stations,fname=event_id+'.json')



