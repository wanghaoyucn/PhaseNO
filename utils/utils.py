import os

import numpy as np
import pandas as pd
import json
from datetime import datetime, timedelta

def slide_window(data, window_size, step_size, axis=0, drop_last=False):
    '''
    yield windowed data with specified window size and step size
    Args:
        data: numpy array
        window_size: int
        step_size: int
        axis: int
        drop_last: bool, if True, drop the last window if it is smaller than window_size
    '''
    n_points = data.shape[axis]
    for i in range(0, n_points-window_size+1, step_size):
        if drop_last and i+window_size > n_points:
            break
        else:
            idx = min(i, n_points-window_size)
        yield data.take(range(idx, idx+window_size), axis=axis, mode='clip'), idx

def save_picks(picks, output_dir, dt=0.01, amps=None, fname=None):
    if fname is None:
        fname = "picks.csv"

    int2s = lambda x: ",".join(["["+",".join(map(str, i))+"]" for i in x])
    flt2s = lambda x: ",".join(["["+",".join(map("{:0.3f}".format, i))+"]" for i in x])
    sci2s = lambda x: ",".join(["["+",".join(map("{:0.3e}".format, i))+"]" for i in x])
    
    if len(picks) == 0:
        with open(os.path.join(output_dir, fname), "w") as fp:
            pass
    picks_ = []
    if amps is None:
        for pick in picks:
            for idx, prob in zip(pick.p_idx, pick.p_prob):
                picks_.append(pd.DataFrame({"station_id": pick.station_id, 
                            "phase_index": pick.p_idx,
                            "phase_time":calc_timestamp(pick.t0, float(idx)*dt), 
                            "phase_score": prob.astype(float), 
                            "phase_amplitude": '',
                            "phase_type": "P"}, index=[0]))
            for idx, prob in zip(pick.s_idx, pick.s_prob):
                picks_.append(pd.DataFrame({"station_id": pick.station_id, 
                            "phase_index": pick.s_idx,
                            "phase_time":calc_timestamp(pick.t0, float(idx)*dt), 
                            "phase_score": prob.astype(float), 
                            "phase_amplitude": '',
                            "phase_type": "S"}, index=[0]))
    else:
        for pick, amplitude in zip(picks, amps):
            for idx, prob, amp in zip(pick.p_idx, pick.p_prob, amplitude.p_amp[0]):
                picks_.append(pd.DataFrame({"station_id": pick.station_id, 
                            "phase_index": pick.p_idx,
                            "phase_time":calc_timestamp(pick.t0, float(idx)*dt), 
                            "phase_score": prob.astype(float), 
                            "phase_amplitude": amp.astype(float),
                            "phase_type": "P"}, index=[0]))
            for idx, prob, amp in zip(pick.s_idx, pick.s_prob, amplitude.s_amp[0]):
                picks_.append(pd.DataFrame({"station_id": pick.station_id, 
                            "phase_index": pick.s_idx,
                            "phase_time":calc_timestamp(pick.t0, float(idx)*dt), 
                            "phase_score": prob.astype(float), 
                            "phase_amplitude": amp.astype(float),
                            "phase_type": "S"}, index=[0]))
    picks_ = pd.concat(picks_, ignore_index=True)
    picks_.to_csv(os.path.join(output_dir, fname), index=False)

    return 0

def calc_timestamp(timestamp, sec):
    timestamp = datetime.strptime(timestamp, "%Y-%m-%dT%H:%M:%S.%f") + timedelta(seconds=sec)
    return timestamp.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]

def save_picks_json(picks, output_dir, dt=0.01, amps=None, fname=None):
    if fname is None:
        fname = "picks.json"

    picks_ = []
    if amps is None:
        for pick in picks:
            for idx, prob in zip(pick.p_idx, pick.p_prob):
                picks_.append({"id": pick.fname, 
                            "timestamp":calc_timestamp(pick.t0, float(idx)*dt), 
                            "prob": prob.astype(float), 
                            "type": "p"})
            for idx, prob in zip(pick.s_idx, pick.s_prob):
                picks_.append({"id": pick.fname, 
                            "timestamp":calc_timestamp(pick.t0, float(idx)*dt), 
                            "prob": prob.astype(float), 
                            "type": "s"})
    else:
        for pick, amplitude in zip(picks, amps):
            for idx, prob, amp in zip(pick.p_idx, pick.p_prob, amplitude.p_amp[0]):
                picks_.append({"id": pick.fname,
                            "timestamp":calc_timestamp(pick.t0, float(idx)*dt), 
                            "prob": prob.astype(float), 
                            "amp": amp.astype(float),
                            "type": "p"})
            for idx, prob, amp in zip(pick.s_idx, pick.s_prob, amplitude.s_amp[0]):
                picks_.append({"id": pick.fname, 
                            "timestamp":calc_timestamp(pick.t0, float(idx)*dt), 
                            "prob": prob.astype(float), 
                            "amp": amp.astype(float),
                            "type": "s"})
                
    with open(os.path.join(output_dir, fname), "w") as fp:
        json.dump(picks_, fp)

    return 0