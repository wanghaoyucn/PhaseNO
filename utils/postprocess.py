from collections import namedtuple

import numpy as np


def _trim_nan(x):
        """
        Removes all starting and trailing nan values from a 1D array and returns the new array and the number of NaNs
        removed per side.
        """
        mask_forward = np.cumprod(np.isnan(x)).astype(
            bool
        )  # cumprod will be one until the first non-Nan value
        x = x[~mask_forward]
        mask_backward = np.cumprod(np.isnan(x)[::-1])[::-1].astype(
            bool
        )  # Double reverse for a backwards cumprod
        x = x[~mask_backward]

        return x, np.sum(mask_forward.astype(int)), np.sum(mask_backward.astype(int))


# codes for picking
# clone from PhaseNet: https://github.com/AI4EPS/PhaseNet
def _detect_peaks(x, mph=None, mpd=1, threshold=0, edge='rising', kpsh=False, valley=False):

    """
    __author__ = "Marcos Duarte, https://github.com/demotu"
    __version__ = "1.0.6"
    __license__ = "MIT"

    Detect peaks in data based on their amplitude and other features.

    Parameters
    ----------
    x : 1D array_like
        data.
        
    mph : {None, number}, default=None
        detect peaks that are greater than minimum peak height.
        
    mpd : int, default=1
        detect peaks that are at least separated by minimum peak distance (in number of data).
        
    threshold : int, default=0
        detect peaks (valleys) that are greater (smaller) than `threshold in relation to their immediate neighbors.
        
    edge : str, default=rising
        for a flat peak, keep only the rising edge ('rising'), only the falling edge ('falling'), both edges ('both'), or don't detect a flat peak (None).
        
    kpsh : bool, default=False
        keep peaks with same height even if they are closer than `mpd`.
        
    valley : bool, default=False
        if True (1), detect valleys (local minima) instead of peaks.

    Returns
    -------
    ind : 1D array_like
        indeces of the peaks in `x`.

    Modified from 
    ----------
    .. [1] http://nbviewer.ipython.org/github/demotu/BMC/blob/master/notebooks/DetectPeaks.ipynb
    

    """

    x = np.atleast_1d(x).astype('float64')
    if x.size < 3:
        return np.array([], dtype=int)
    if valley:
        x = -x
    # find indices of all peaks
    dx = x[1:] - x[:-1]
    # handle NaN's
    indnan = np.where(np.isnan(x))[0]
    if indnan.size:
        x[indnan] = np.inf
        dx[np.where(np.isnan(dx))[0]] = np.inf
    ine, ire, ife = np.array([[], [], []], dtype=int)
    if not edge:
        ine = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) > 0))[0]
    else:
        if edge.lower() in ['rising', 'both']:
            ire = np.where((np.hstack((dx, 0)) <= 0) & (np.hstack((0, dx)) > 0))[0]
        if edge.lower() in ['falling', 'both']:
            ife = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) >= 0))[0]
    ind = np.unique(np.hstack((ine, ire, ife)))
    # handle NaN's
    if ind.size and indnan.size:
        # NaN's and values close to NaN's cannot be peaks
        ind = ind[np.in1d(ind, np.unique(np.hstack((indnan, indnan-1, indnan+1))), invert=True)]
    # first and last values of x cannot be peaks
    if ind.size and ind[0] == 0:
        ind = ind[1:]
    if ind.size and ind[-1] == x.size-1:
        ind = ind[:-1]
    # remove peaks < minimum peak height
    if ind.size and mph is not None:
        ind = ind[x[ind] >= mph]
    # remove peaks - neighbors < threshold
    if ind.size and threshold > 0:
        dx = np.min(np.vstack([x[ind]-x[ind-1], x[ind]-x[ind+1]]), axis=0)
        ind = np.delete(ind, np.where(dx < threshold)[0])
    # detect small peaks closer than minimum peak distance
    if ind.size and mpd > 1:
        ind = ind[np.argsort(x[ind])][::-1]  # sort ind by peak height
        idel = np.zeros(ind.size, dtype=bool)
        for i in range(ind.size):
            if not idel[i]:
                # keep peaks with the same height if kpsh is True
                idel = idel | (ind >= ind[i] - mpd) & (ind <= ind[i] + mpd) \
                    & (x[ind[i]] > x[ind] if kpsh else True)
                idel[i] = 0  # Keep current peak
        # remove the small peaks and sort back the indices by their occurrence
        ind = np.sort(ind[~idel])

    return ind, x[ind]

def extract_amplitude(data, picks, window_p=10, window_s=5, dt=0.01):
    record = namedtuple("amplitude", ["p_amp", "s_amp"])
    window_p = int(window_p/dt)
    window_s = int(window_s/dt)
    amps = []; 
    for i, (da, pi) in enumerate(zip(data, picks)):
        p_amp, s_amp = [], []

        amp = np.max(np.abs(da), axis=0)
        #amp = np.median(np.abs(da[:,j,:]), axis=-1)
        #amp = np.linalg.norm(da[:,j,:], axis=-1)
        tmp = []
        for k in range(len(pi.p_idx)-1):
            tmp.append(np.max(amp[pi.p_idx[k]:min(pi.p_idx[k]+window_p, pi.p_idx[k+1])]))
        if len(pi.p_idx) >= 1:
            try:
                tmp.append(np.max(amp[pi.p_idx[-1]:pi.p_idx[-1]+window_p]))
            except ValueError:  #raised if `y` is empty.
                print('P pick index is outside of data index! index = ', pi.p_idx[-1])
                tmp.append(np.float32(0))

        p_amp.append(tmp)
        tmp = []
        for k in range(len(pi.s_idx)-1):
            tmp.append(np.max(amp[pi.s_idx[k]:min(pi.s_idx[k]+window_s, pi.s_idx[k+1])]))
        if len(pi.s_idx) >= 1:
            try:
                tmp.append(np.max(amp[pi.s_idx[-1]:pi.s_idx[-1]+window_s]))
            except ValueError:
                print('S pick index is outside of data index! index = ', pi.s_idx[-1])
                tmp.append(np.float32(0))

        s_amp.append(tmp)
        amps.append(record(p_amp, s_amp))
    return amps