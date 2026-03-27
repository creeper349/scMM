import numpy as np
import os
import json
import pandas as pd
import logging
from joblib import Parallel, delayed

from .io import (load_single_file, 
                 sum_spec, 
                 extract_peaks, 
                 align_frame, 
                 sum_spectrum_from_file, 
                 pack_specs)
from ..util.peak import peak_detection_recon, peak_profiling
from ..util.normalize import normalize

from typing import Callable, Optional, Dict, Any, Self, Literal, Hashable
from scipy.ndimage import median_filter, grey_opening
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.ensemble import IsolationForest
from datetime import datetime

DebugHook = Callable[[str, Dict[str, Any]], None]

def _align_frame_from_file(
    file_path: str,
    mz_list,
    ppm_tol: int = 10,
    dtype=np.float64
):
    exp, file_meta = load_single_file(file_path, format="auto")
    data, peak_meta = align_frame(exp, mz_list, ppm_tol, dtype=dtype)
    return {
        "file_meta": file_meta,
        "data": data,
        "peak_meta": peak_meta,
    }

class CyESIData:
    def __init__(self, result_dir:str, dtype = np.float64):
        with open(os.path.join(result_dir, ".meta"), 'r') as fp:
            file_meta = json.load(fp)

        data_path_pkl = os.path.join(result_dir, "data.pkl")
        peak_path_pkl = os.path.join(result_dir, "peak_profile.pkl")
        data = None
        peak_meta = None
        if os.path.exists(data_path_pkl):
            data = pd.read_pickle(data_path_pkl)
            if os.path.exists(peak_path_pkl):
                peak_meta = pd.read_pickle(peak_path_pkl)
        else:
            data_path_csv = os.path.join(result_dir, "data.csv")
            peak_path_csv = os.path.join(result_dir, "peak_profile.csv")
            if os.path.exists(data_path_csv):
                data = pd.read_csv(data_path_csv, index_col=0, dtype=dtype)
            if os.path.exists(peak_path_csv):
                peak_meta = pd.read_csv(peak_path_csv, index_col=0, dtype=dtype)

        if data is None:
            raise FileNotFoundError(f"No processed data found in {result_dir}")
        self.data = data
        self.peak_meta = peak_meta
        self.file_meta = file_meta
        
    @classmethod
    def load_from_file(cls, file_path:str,
                    ref_mz: Optional[float] = None, 
                    dtype = np.float64,
                    ppm_tol: int = 10,
                    prominence_ratio: float = None,
                    distance:int = 3):
        obj = object.__new__(cls)
        exp, obj.file_meta = load_single_file(file_path, format='auto')
        sum_ = sum_spec(exp)
        obj.file_meta["ref_mz"] = ref_mz
        
        mz_list = extract_peaks(0, sum_, dtype=dtype, prominence_ratio=prominence_ratio, distance=distance)
        obj.data, obj.peak_meta = align_frame(exp, mz_list, ppm_tol, dtype=dtype)
        obj.peak_meta["time"] = obj.peak_meta["rt"] / np.max(obj.peak_meta["rt"])
        obj.ref_mz = ref_mz
        obj.preprocess()
        return obj
        
    @classmethod
    def load_from_filelist(cls, dir_path:str,
                    ref_mz: Optional[float] = None,
                    dtype = np.float64,
                    ppm_tol: int = 10,
                    prominence_ratio: float = None,
                    n_jobs:int = -1,
                    distance:int = 3):
        files = os.listdir(dir_path)
        filelist = []
        for file in files:
            full_path = os.path.join(dir_path, file)
            if (not os.path.isdir(full_path)) and file.lower().endswith((".mzml", ".mzxml")):
                filelist.append(full_path)
        _sum_specs = Parallel(n_jobs=n_jobs, prefer="threads")(delayed(sum_spectrum_from_file)(f) for f in filelist)
        total_sum_spec = sum_spec(pack_specs(_sum_specs))
        _, _, mz_list, _ = extract_peaks(0, total_sum_spec, dtype=dtype, prominence_ratio=prominence_ratio, distance=distance)
        align_results = Parallel(n_jobs=n_jobs)(
            delayed(_align_frame_from_file)(
                fp, mz_list, ppm_tol=ppm_tol, dtype=dtype
            )
            for fp in filelist
        )
        
        obj = cls.__new__(cls)
        obj.data = None
        obj.peak_meta = None
        timestamp_start = align_results[0]["file_meta"]["timestamp"]
        timestamp_end = align_results[-1]["file_meta"]["timestamp"] + \
            align_results[-1]["peak_meta"]["rt"].iloc[-1]

        per_file_meta = []

        for r in align_results:
            data = r["data"]
            peak_meta = r["peak_meta"]
            file_meta = dict(r["file_meta"])

            if isinstance(data, pd.DataFrame):
                data = data.copy()
                peak_meta["time"] = (peak_meta["rt"] + file_meta["timestamp"] - timestamp_start
                                     ) / (timestamp_end - timestamp_start)

                if obj.data is None:
                    obj.data = data
                else:
                    obj.data = pd.concat([obj.data, data], axis=0)

            if isinstance(peak_meta, pd.DataFrame):
                peak_meta = peak_meta.copy()

                if obj.peak_meta is None:
                    obj.peak_meta = peak_meta
                else:
                    obj.peak_meta = pd.concat([obj.peak_meta, peak_meta], axis=0)

            per_file_meta.append(file_meta)

        obj.file_meta = {
            "name": os.path.basename(os.path.normpath(dir_path)),
            "ref_mz": ref_mz,
            "per_file_meta": per_file_meta,
        }
        obj.ref_mz = ref_mz

        obj.preprocess()
        return obj
        
    def preprocess(self, baseline_filter = grey_opening, 
                         baseline_filter_size:int = 15,
                         peak_lam:float = 0.5, 
                         peak_sigma_min:float = 1e-3, 
                         tau:float = 2,
                         zero_threshold:float = 0.9,
                         peak_profiles: list = ['rt', 'width', 'symmetry'],
                         subtract_baseline: bool = False,
                         n_jobs:int = -1, 
                         debug_hook: Optional[DebugHook] = None,
                         **kwargs):
        
        def emit(stage: str, **payload):
            if debug_hook is not None:
                debug_hook(stage, payload)
                
        cell_mask, peak_mask, C, B, sigma, r1 = peak_detection_recon(self.data, 
                                                    baseline_filter=baseline_filter,
                                                    baseline_filter_size=baseline_filter_size,
                                                    ref_mz=self.ref_mz,
                                                    peak_lam=peak_lam,
                                                    peak_sigma_min=peak_sigma_min,
                                                    tau=tau,
                                                    n_jobs=n_jobs,
                                                    dtype=self.data.values.dtype,
                                                    **kwargs)
        emit("peak_detection", data = self.data, cell_mask=cell_mask, ref_mz=self.ref_mz)
        emit("cell_signal", C = C, data = self.data, ref_mz=self.ref_mz)
        emit("r1", r1 = r1)
        self.data, self.peak_meta = peak_profiling(self.data, B, cell_mask, peak_mask, 
                                                   time = self.peak_meta["rt"].values, 
                                                   ref_mz=self.ref_mz, 
                                                   profiling = peak_profiles,
                                                   subtract_baseline = subtract_baseline,
                                                   dtype=self.data.values.dtype)
        include_columns = (self.data.values > 0).mean(axis = 0) > 1 - zero_threshold
        self.data = self.data.iloc[:, include_columns]
        self.file_meta['length'] = self.data.shape[0]
        self._process_flag = True
        return self
            
    def impute(self, method:str = 'knn', missing_values = 0, **kwargs):
        logging.info(f"Run data imputing on {self.get_name()}, method:{method}")
        if method == 'knn':
            imputer = KNNImputer(missing_values=missing_values, **kwargs)
        else:
            imputer = SimpleImputer(missing_values=missing_values, strategy=method, **kwargs)
            
        self.data = pd.DataFrame(
            imputer.fit_transform(self.data),
            columns = self.data.columns,
            dtype = self.data.values.dtype
        )
        return self
    
    def remove_outlier(self, **kwargs):
        iso = IsolationForest(**kwargs)
        inlier_id = (iso.fit_predict(self.data) == 1)
        self.data = self.data.iloc[inlier_id, :]
        self.peak_meta = self.peak_meta.iloc[inlier_id, :]
        return self
                    
    def normalize(self, method:str = "total", **norm_kwargs):
        logging.info(f"Run normalization on {self.get_name()}, method:{method}")
        self.data = pd.DataFrame(
            normalize(self.data.values, method, norm_kwargs),
            columns = self.data.columns,
            dtype = self.data.values.dtype
        )
        return self
    
    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, key):
        key = float(key)
        idx = (np.abs(self.data.columns.values.astype(float) - key)).argmin()
        return self.data.iloc[:, idx].values
    
    def save(self, root_path:str):
        dir_name = os.path.join(root_path, self.file_meta["name"])
        os.mkdir(dir_name)
        with open(os.path.join(dir_name, ".meta"), 'w') as fp:
            file_meta = json.dump(self.file_meta, fp)
        self.data.to_csv(os.path.join(dir_name, "data.csv"))
        self.peak_meta.to_csv(os.path.join(dir_name, "peak_meta.csv"))
        
    def to_anndata(self):
        obs_df = pd.DataFrame({
            "rt": self.peak_meta["rt"],
            "time": self.peak_meta["time"],
            "width": self.peak_meta["width"],
            "symmetry": self.peak_meta["symmetry"]
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