"""Verbatim _nrmse_1d from user util.py; original saved in references/user_nrmse.
Source SHA256: 0b507455befa6a630dc610127fe2f264fa635dab73bb055df5484478ce938741
"""
import numpy as np


def _nrmse_1d(A, B, *, norm="rms"):
    """
    Normalized RMSE:
      - 'rms'     : RMSE(A,B) / RMS(A)        = RMSE / sqrt(mean(A**2))
      - 'min-max' : RMSE(A,B) / (max(A,B)-min(A,B))
      - 'std'     : RMSE(A,B) / std(A)
      - 'mean'    : RMSE(A,B) / mean(|A|)
    """
    # The data that serves as the basis for comparison is A
    A = np.asarray(A, np.float64); B = np.asarray(B, np.float64)
    dmin = float(min(A.min(), B.min())); dmax = float(max(A.max(), B.max()))
    data_range = dmax - dmin
    if data_range < 0:
        raise ValueError("data range is strange!!")
    elif data_range == 0:
        return 0.0
        
    mse = np.mean((A - B)**2)
    rmse = float(np.sqrt(mse))
    if norm == "rms":
        denom = float(np.sqrt(np.mean(A**2)))
    elif norm == "min-max":
        dmin = float(min(A.min(), B.min())); dmax = float(max(A.max(), B.max()))
        denom = dmax - dmin
    elif norm == "std":
        denom = float(np.std(A))
    elif norm == "mean":
        denom = float(np.mean(np.abs(A)))
    else:
        raise ValueError("norm must be 'rms', 'min-max', 'std', or 'mean'")

    if denom == 0:
        raise ValueError("ref data value is 0")
    return float(rmse / denom)
