import pandas as pd
import numpy as np
from anndata import AnnData
from typing import Optional
from .data import CyESIData

def to_anndata(data:CyESIData):
    obs_df = pd.DataFrame({
        "cell_id": data.data.index,
        "labels": data.get_labels(),
        "time": data.get_time(),
        "width": data.peak_meta['width'].values,
        "symmetry": data.peak_meta['symmetry'].values
    })
    
    var_df = pd.DataFrame({
        "mz": data.data.columns
    })
    
    adata = AnnData(
        X=data.data.values,
        obs=obs_df.set_index("cell_id"),
        var=var_df.set_index("mz")
    )
    adata.raw = adata.copy()
    return adata

class MetaboData(AnnData):
    def __init__(self, 
                 X: np.ndarray,
                 obs: Optional[pd.DataFrame] = None,
                 var: Optional[pd.DataFrame] = None,
                 is_inten: Optional[pd.DataFrame] = None,
                 **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
        super().__init__(X=X, obs=obs, var=var, **kwargs)
        assert (is_inten is None) or (is_inten.shape[0] == X.shape[0])
        self.is_inten = is_inten
        
    @property
    def n_observations(self):
            return self.X.shape[0]
    @property
    def n_features(self):
            return self.X.shape[1]
        
    def is_calibration(self, reg:float = 0):
        log_is = np.log1p(self.is_inten.values)
        is_central = log_is - log_is.mean(axis=0)
        proj_is = is_central @ \
            np.linalg.inv(is_central.T @ is_central + reg * np.eye(is_central.shape[1]))\
            @ is_central.T
        self.layers["X_calibrated"] = (np.eye(proj_is.shape[0]) - proj_is) @ np.log1p(self.X)
        self.X = self.layers["X_calibrated"]
        self.uns["X_rep"] = "X_calibrated"
        return self
    
    def mz_calibration(self, mz_is_true: list|np.ndarray, 
                       mz_is_obs: list|np.ndarray = None):
        assert "mz" in self.var.columns
        mz_obs_is = np.asarray(mz_is_obs, dtype=float) if mz_is_obs is not None\
            else self.is_inten.columns.astype(float)
        mz_theory_is = np.asarray(mz_is_true, dtype=float)
        
        if mz_obs_is.shape != mz_theory_is.shape:
            raise ValueError("Observed and theoretical m/z must have same shape")
        
        A = np.vstack([mz_theory_is, np.ones_like(mz_theory_is)]).T
        a, b = np.linalg.lstsq(A, mz_obs_is, rcond=None)[0]
        
        mz_obs_all = self.var["mz"].values.astype(float)
        mz_calibrated = (mz_obs_all - b) / a
        self.var["mz_calibrated"] = mz_calibrated
        self.uns["mz_calibration_params"] = {"a": a, "b": b,
            "ppm_res": (mz_obs_is - (a * mz_theory_is + b)) / mz_theory_is * 1e6}
        return self