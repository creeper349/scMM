import numpy as np
import os
import json
import pandas as pd
import logging
from joblib import Parallel, delayed
from anndata import AnnData

from .io import (load_single_file, 
                 sum_spec, 
                 extract_peaks, 
                 align_frame, 
                 sum_spectrum_from_file, 
                 pack_specs)
from ..util.peak import filter_spectrum, find_cell_peaks
from ..util.normalize import normalize

from typing import Callable, Optional, Dict, Any, Self, Literal, Hashable
from scipy.ndimage import median_filter, grey_opening
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.ensemble import IsolationForest

DebugHook = Callable[[str, Dict[str, Any]], None]

def _align_frame_from_file(
    file_path: str,
    mz_list,
    ppm_tol: int = 10,
    dtype=np.float64
):
    exp, file_meta = load_single_file(file_path, format="auto")
    logging.info(f"Aligning frames from MS file {file_path}...")
    data, peak_meta = align_frame(exp, mz_list, ppm_tol, dtype=dtype)
    return {
        "file_meta": file_meta,
        "data": data,
        "peak_meta": peak_meta,
    }

class CyESIData:
    def __init__(self, result_dir:str):
        with open(os.path.join(result_dir, ".meta"), 'r') as fp:
            file_meta = json.load(fp)

        data_path_pkl = os.path.join(result_dir, "data.pkl")
        peak_path_pkl = os.path.join(result_dir, "peak_meta.pkl")
        data = None
        peak_meta = None
        if os.path.exists(data_path_pkl):
            data = pd.read_pickle(data_path_pkl)
            if os.path.exists(peak_path_pkl):
                peak_meta = pd.read_pickle(peak_path_pkl)
        else:
            data_path_csv = os.path.join(result_dir, "data.csv")
            peak_path_csv = os.path.join(result_dir, "peak_meta.csv")
            if os.path.exists(data_path_csv):
                data = pd.read_csv(data_path_csv, index_col=0)
            if os.path.exists(peak_path_csv):
                peak_meta = pd.read_csv(peak_path_csv, index_col=0)

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
                    resolution: float = 35000,
                    resample_points_per_fwhm: float = 5.0,
                    ms_peak_snr_threshold: float = 10.0,
                    prominence_ratio: float = None,
                    distance:int = 3,
                    **preprocess_kwds):
        obj = object.__new__(cls)
        exp, obj.file_meta = load_single_file(file_path, format='auto')
        sum_ = sum_spec(exp, resolution_200=resolution, points_per_fwhm=resample_points_per_fwhm)
        obj.file_meta["ref_mz"] = ref_mz
        
        sum_ = filter_spectrum(sum_, snr_threshold=ms_peak_snr_threshold)
        mz_list, _ = extract_peaks(sum_, dtype=dtype, prominence_ratio=prominence_ratio, distance=distance)
        obj.data, obj.peak_meta = align_frame(exp, mz_list, ppm_tol, dtype=dtype)
        obj.peak_meta["time"] = obj.peak_meta["rt"] / np.max(obj.peak_meta["rt"])
        obj.peak_meta["label"] = [obj.file_meta["name"].split(".")[0]] * len(obj.peak_meta)
        obj.ref_mz = ref_mz
        obj.preprocess(**preprocess_kwds)
        return obj
        
    @classmethod
    def load_from_filelist(cls, dir_path:str,
                    ref_mz: Optional[float] = None,
                    dtype = np.float64,
                    ppm_tol: int = 10,
                    resolution: float = 35000,
                    resample_points_per_fwhm: float = 5.0,
                    ms_peak_snr_threshold: float = 10.0,
                    prominence_ratio: float = None,
                    n_jobs:int = -1,
                    distance:int = 3,
                    **preprocess_kwds):
        files = os.listdir(dir_path)
        filelist = []
        for file in files:
            full_path = os.path.join(dir_path, file)
            if (not os.path.isdir(full_path)) and file.lower().endswith((".mzml", ".mzxml")):
                filelist.append(full_path)
        logging.info(f"Detected files in targeted directory: {filelist}")
        logging.info(f"Summing MS Spectrometry, resolution={resolution}, resample points per FWHM={resample_points_per_fwhm}...")
        _sum_specs = Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(sum_spectrum_from_file)(f, resolution_200=resolution, points_per_fwhm=resample_points_per_fwhm)
            for f in filelist)
        total_sum_spec = sum_spec(pack_specs(_sum_specs), resolution_200=resolution, points_per_fwhm=resample_points_per_fwhm)
        
        logging.info(f"Performing summed-MS denoising, snr_threshold={ms_peak_snr_threshold}...")
        total_sum_spec = filter_spectrum(total_sum_spec, snr_threshold=ms_peak_snr_threshold)
        logging.info(f"Performing summed-MS peak picking...")
        mz_list, _ = extract_peaks(total_sum_spec, prominence_ratio=prominence_ratio, distance=distance)
        align_results = Parallel(n_jobs=n_jobs)(
            delayed(_align_frame_from_file)(
                fp, mz_list, ppm_tol=ppm_tol, dtype=dtype
            )
            for fp in filelist
        )
        
        logging.info(f"Building data container class...")
        align_results = sorted(align_results,key=lambda x: x["file_meta"]["timestamp"])
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
                peak_meta["label"] = [file_meta["name"].split(".")[0]] * len(peak_meta)
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
        
        logging.info(f"Performing denoising and cell peak picking...")
        obj.preprocess(**preprocess_kwds)
        return obj
        
    def preprocess(self, baseline_filter = median_filter, 
                         baseline_filter_size:int = 50,
                         cell_snr:float = 5.0,
                         peak_snr:float = 3.0,
                         max_zero_frac:float = 0.9,
                         debug_hook: Optional[DebugHook] = None,
                         **kwargs):
        
        def emit(stage: str, **payload):
            if debug_hook is not None:
                debug_hook(stage, payload)
                
        data = find_cell_peaks(self.data, self.ref_mz, baseline_filter=baseline_filter, baseline_filter_size=baseline_filter_size, 
                               cell_snr=cell_snr, peak_snr=peak_snr, max_zero_frac=max_zero_frac, **kwargs)
        emit("find_cells", signal = self.data, baseline = data["baseline"], cell_idx = data["peak_frames"])
        self.data, self.peak_meta = (data["cell_df"], 
            pd.DataFrame(self.peak_meta.iloc[data["peak_frames"], :], index=self.peak_meta.index[data["peak_frames"]]))
        self.file_meta['length'] = self.data.shape[0]
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
        logging.info(f"Run normalization on {self.file_meta['name']}, method:{method}")
        self.data = pd.DataFrame(
            normalize(self.data.values, method, norm_kwargs),
            columns = self.data.columns,
            dtype = self.data.values.dtype
        )
        return self
    
    def alignwith(self, other:Self, ppm_tol:float = 5.0, mz_merge_options: Literal["union", "ref"] = "union"):
        
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

        if mz_merge_options == "union":
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
        per_file_meta = []
        if "per_file_meta" in self.file_meta:
            if "per_file_meta" not in other.file_meta:
                per_file_meta.append(other.file_meta)
            else:
                per_file_meta.extend(other.file_meta.get("per_file_meta", []))
        else:
            per_file_meta = [self.file_meta, other.file_meta]
        self.file_meta = {
            "name": f"{self.file_meta['name']}+{other.file_meta['name']}",
            "per_file_meta": per_file_meta
        }

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
        logging.info(f"Saving processed data to {dir_name}...")
        with open(os.path.join(dir_name, ".meta"), 'w') as fp:
            file_meta = json.dump(self.file_meta, fp)
        self.data.to_csv(os.path.join(dir_name, "data.csv"))
        self.peak_meta.to_csv(os.path.join(dir_name, "peak_meta.csv"))
        
    def to_anndata(self):
        obs_df = pd.DataFrame({
            "cell_id": self.peak_meta.index,
        })
        for col in self.peak_meta.columns:
            obs_df[col] = self.peak_meta[col].values
        
        var_df = pd.DataFrame({
            "mz": self.data.columns
        })
        
        adata = AnnData(
            X=self.data.values,
            obs=obs_df.set_index("cell_id"),
            var=var_df
        )
        adata.raw = adata.copy()
        return adata