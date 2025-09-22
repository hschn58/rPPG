import numpy as np

def bandpass(x, fs, lo, hi):
    # replace with your real function when ready
    return x

def test_bandpass_identity_when_full_band():
    fs = 30.0
    t = np.arange(0, 3, 1/fs)
    x = np.sin(2*np.pi*1.0*t) + 0.1*np.random.RandomState(0).randn(t.size)
    y = bandpass(x, fs, lo=0.1, hi=fs/2-0.1)
    assert len(y) == len(x)
