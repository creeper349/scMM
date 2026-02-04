import numpy as np
import os
import json
import pandas as pd
import logging

from .io import load_file, frame_concat
from ..util.peak import peak_detection_recon, peak_profiling
from ..util.normalize import normalize

from typing import Callable, Optional, Dict, Any, Self, Literal, Hashable
from scipy.ndimage import median_filter, grey_opening
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.ensemble import IsolationForest
from datetime import datetime

DebugHook = Callable[[str, Dict[str, Any]], None]

class CyESIData:
    def __init__(self, file_path, 
                 ref_mz: Optional[float] = None, 
                 dtype = np.float64,
                 ppm_tol: int = 10,
                 zero_filter: float = 0.99):
        exp, self.file_meta = load_file(file_path, format='auto')
        self.file_meta["ref_mz"] = ref_mz
        self.data, self.peak_meta = frame_concat(exp, ppm_tol=ppm_tol, zero_filter=zero_filter, dtype=dtype)
        self.ref_mz = ref_mz
        self._process_flag = False
        self._concat_flag = False
        
    @classmethod
    def load_from_processed(cls, result_dir:str, dtype = np.float64):
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
                data = pd.read_csv(data_path_csv, index_col=0)
            if os.path.exists(peak_path_csv):
                peak_meta = pd.read_csv(peak_path_csv, index_col=0)

        if data is None:
            raise FileNotFoundError(f"No processed data found in {result_dir}")

        obj = object.__new__(cls)
        obj.file_meta = file_meta
        try:
            obj.data = data.astype(dtype)
        except Exception:
            obj.data = data
        obj.peak_meta = peak_meta
        if isinstance(file_meta, dict):
            obj.ref_mz = file_meta.get("ref_mz", None)
            obj.file_meta['length'] = data.shape[0]
        else:
            obj.ref_mz = None
        obj._process_flag = True
        obj._concat_flag = isinstance(file_meta, list)
        return obj
    
    @classmethod
    def load_from_df(cls, data:pd.DataFrame, 
                     peak_meta:pd.DataFrame, 
                     file_meta:Optional[Hashable] = None,
                     dtype = np.float64):
        obj = object.__new__(cls)
        obj.file_meta = file_meta
        try:
            obj.data = data.astype(dtype)
        except Exception:
            obj.data = data
        obj.peak_meta = peak_meta
        if isinstance(file_meta, dict):
            obj.ref_mz = file_meta.get("ref_mz", None)
            obj.file_meta['length'] = data.shape[0]
        else:
            obj.ref_mz = None
        obj._process_flag = True
        obj._concat_flag = isinstance(file_meta, list)
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
                                                   time = self.peak_meta["RT"], 
                                                   ref_mz=self.ref_mz, 
                                                   profiling = peak_profiles,
                                                   subtract_baseline = subtract_baseline,
                                                   dtype=self.data.values.dtype)
        include_columns = (self.data.values > 0).mean(axis = 0) > 1 - zero_threshold
        self.data = self.data.iloc[:, include_columns]
        self.file_meta['length'] = self.data.shape[0]
        self._process_flag = True
        return self
        
    def save(self, result_dir:str = None, binary:bool = False):
        if result_dir is None: result_dir = os.getcwd()
        exp_name = self.file_meta[0]["name"] if self._concat_flag else self.file_meta["name"]
        result_dir = os.path.join(result_dir, exp_name)
        os.makedirs(result_dir, exist_ok=True)
        with open(os.path.join(result_dir, ".meta"), mode='w') as fp:
            json.dump(self.file_meta, fp, indent=2)
        if binary:
            self.data.to_pickle(os.path.join(result_dir, "data.pkl"))
            self.peak_meta.to_pickle(os.path.join(result_dir, "peak_profile.pkl"))
        else:
            self.data.to_csv(os.path.join(result_dir, "data.csv"))
            self.peak_meta.to_csv(os.path.join(result_dir, "peak_profile.csv"))
            
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
        self.data = self.data[inlier_id, :]
        self.peak_meta = self.peak_meta[inlier_id, :]
        return self
    
    def alignwith(self, other:Self, ppm_tol:int = 10, unmatched:Literal['del', 'pad'] = 'del'):
        assert(self._process_flag and other._process_flag)
        
        df1, df2 = self.data, other.data
        mz1, mz2 = df1.columns.values.astype(self.data.values.dtype), df2.columns.values.astype(self.data.values.dtype)
        idx2_aligned = np.full(len(mz1), -1, dtype=int)

        j = 0
        for i, m in enumerate(mz1):
            while j < len(mz2) and mz2[j] < m * (1 - ppm_tol * 1e-6):
                j += 1
            if j < len(mz2) and abs(mz2[j] - m) / m * 1e6 <= ppm_tol:
                idx2_aligned[i] = j

        keep = idx2_aligned >= 0
        mz_aligned = df1.columns[keep]
        df1_aligned = df1.loc[:, mz_aligned]
        df2_aligned = df2.iloc[:, idx2_aligned[keep]]
        df2_aligned.columns = mz_aligned

        merged_df = pd.concat([df1_aligned, df2_aligned], axis=0, ignore_index=True)

        if unmatched == 'pad':
            mask_new = np.ones(len(mz2), dtype=bool)
            mask_new[idx2_aligned[keep]] = False
            new_mz = mz2[mask_new]
            if len(new_mz) > 0:
                df2_new = df2.loc[:, new_mz].copy()
                df2_new[:] = 0
                merged_df = pd.concat([merged_df, df2_new], axis=1)
                
        self.data = merged_df
        self.peak_meta = pd.concat([self.peak_meta, other.peak_meta], axis = 0, ignore_index=True)
        if not self._concat_flag:
            self.file_meta = [self.file_meta]
        self.file_meta.append(other.file_meta)
        self._concat_flag = True

        return self
    
    def bootstrap(self, n_subsamples:int, n_samples:Optional[int] = None, random_state = 42):
        rng = np.random.default_rng(random_state)
        n = self.data.shape[0]
        if n_samples is None:
            n_samples = n
        
        boot_list = []
        for _ in range(n_subsamples):
            indices = rng.integers(0, n, size=n_samples)
            indices = np.unique(indices)
            subdata = self.data.iloc[indices].reset_index(drop=True)
            subdata_meta = self.peak_meta.iloc[indices].reset_index(drop=True)
            subdata_name = self.file_meta["name"] if isinstance(self.file_meta, dict) else self.file_meta[0]["name"]
            boot_list.append(
                CyESIData.load_from_df(subdata, 
                                       subdata_meta,
                                       file_meta = {
                                           "name": f"boot@{subdata_name}",
                                           "ref_mz": self.ref_mz,
                                           "length": len(indices)
                                       })
            )
        return boot_list
    
    def get_labels(self, mapping:dict = None):
        if ('labels' in self.peak_meta.columns) and (mapping is None):
            return self.peak_meta["labels"]
        else:
            labels = np.empty((self.data.shape[0],), dtype = object)
            if self._concat_flag:
                length = 0
                for meta in self.file_meta:
                    if mapping is None:
                        labels[length: length + meta["length"]] = meta["name"]
                    else:
                        labels[length: length + meta["length"]] = mapping[meta["name"]]
                    length += meta["length"]
            self.peak_meta["labels"] = labels
                
            return labels
    
    def get_name(self):
        return self.file_meta["name"] if isinstance(self.file_meta, dict) else self.file_meta[0]["name"]
    
    def get_time(self):
        if not isinstance(self.file_meta, list):
            file_meta = [self.file_meta]
        else:
            file_meta = self.file_meta
            
        idx = 0
        real_time = np.zeros((self.peak_meta.shape[0], ))
        for meta in file_meta:
            try:
                t = meta['time']
            except Exception:
                logging.warning(f"No time metadata find in {self.get_name()}")
                
            dt = datetime.strptime(t, "%y-%m-%d-%H-%M-%S").timestamp()
            next_len = meta['length']
            rt = self.peak_meta['rt'][idx: idx + next_len].values
            real_time[idx: idx + next_len] = dt + rt
            idx += next_len
            
        real_time = (real_time - real_time.min()) / (real_time.max() - real_time.min())
        self.peak_meta['real_time'] = real_time
        return real_time
                    
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