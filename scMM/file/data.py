import numpy as np
import os
import json
import pandas as pd
import logging
import math
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
from ..util.annotation import SDFMzSearcher, DEFAULT_ADDUCTS_NEG, DEFAULT_ADDUCTS_POS

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
        feature_path_pkl = os.path.join(result_dir, "feature_meta.pkl")
        data = None
        peak_meta = None
        feature_meta = None
        if os.path.exists(data_path_pkl):
            data = pd.read_pickle(data_path_pkl)
            if os.path.exists(peak_path_pkl):
                peak_meta = pd.read_pickle(peak_path_pkl)
            if os.path.exists(feature_path_pkl):
                feature_meta = pd.read_pickle(feature_path_pkl)
        else:
            data_path_csv = os.path.join(result_dir, "data.csv")
            peak_path_csv = os.path.join(result_dir, "peak_meta.csv")
            feature_path_csv = os.path.join(result_dir, "feature_meta.csv")
            if os.path.exists(data_path_csv):
                data = pd.read_csv(data_path_csv, index_col=0)
            if os.path.exists(peak_path_csv):
                peak_meta = pd.read_csv(peak_path_csv, index_col=0)
            if os.path.exists(feature_path_csv):
                feature_meta = pd.read_csv(feature_path_csv, index_col=0)

        if data is None:
            raise FileNotFoundError(f"No processed data found in {result_dir}")
        self.data = data
        self.peak_meta = peak_meta
        self.file_meta = file_meta
        self.feature_meta = feature_meta

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
        obj.feature_meta = pd.DataFrame({
            "mz": obj.data.columns.astype(float)
        })
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
        obj.feature_meta = pd.DataFrame({
            "mz": obj.data.columns.astype(float)
        })
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
        self.feature_meta.to_csv(os.path.join(dir_name, "feature_meta.csv"))
        
    def to_anndata(self):
        obs_df = pd.DataFrame({
            "cell_id": self.peak_meta.index,
        })
        var_df = pd.DataFrame({
            "feature_id": self.feature_meta.index
        })
        for col in self.peak_meta.columns:
            obs_df[col] = self.peak_meta[col].values
        for col in self.feature_meta.columns:
            var_df[col] = self.feature_meta[col].values

        adata = AnnData(
            X=self.data.values,
            obs=obs_df.set_index("cell_id"),
            var=var_df.set_index("feature_id")
        )
        adata.raw = adata.copy()
        return adata
    
    def deisotope(
        self,
        isotope_diff: float = 1.003355,
        ppm_tol: float = 1.0,
        max_isotope_order: int = 3,
        r_square_threshold: float = 0.95,
        carbon13_abundance: float = 0.0109,
        intensity_threshold: float = 0.0,
        safety_factor: float = 1.0,
        missing_func=lambda x: x == 0,
        merge_mode: str = "keep_parent",
        remove: bool = True,
        inplace: bool = True,
    ):

        if merge_mode not in {"keep_parent", "sum"}:
            raise ValueError("merge_mode must be either 'keep_parent' or 'sum'.")

        df = self.data
        mz = df.columns.astype(float).to_numpy()
        n_features = len(mz)

        # ---------- ensure feature_meta ----------
        if not hasattr(self, "feature_meta") or self.feature_meta is None:
            self.feature_meta = pd.DataFrame(index=df.columns)

        self.feature_meta = self.feature_meta.reindex(df.columns)

        if "mz" not in self.feature_meta.columns:
            self.feature_meta["mz"] = mz

        # ---------- Step 1: m/z candidate search ----------
        dmz = mz[None, :] - mz[:, None]

        isotope_order = np.rint(dmz / isotope_diff).astype(int)
        expected_dmz = isotope_order * isotope_diff

        ppm_error = np.abs(dmz - expected_dmz) / mz[:, None] * 1e6

        candidate_mask = (
            (isotope_order >= 1)
            & (isotope_order <= max_isotope_order)
            & (dmz > 0)
            & (ppm_error <= ppm_tol)
        )

        candidate_rows, candidate_cols = np.where(candidate_mask)

        candidate_table = pd.DataFrame({
            "parent_index": candidate_rows,
            "isotope_index": candidate_cols,
            "parent_feature": df.columns[candidate_rows],
            "isotope_feature": df.columns[candidate_cols],
            "parent_mz": mz[candidate_rows],
            "isotope_mz": mz[candidate_cols],
            "isotope_order": isotope_order[candidate_rows, candidate_cols],
            "ppm_error": ppm_error[candidate_rows, candidate_cols],
        })

        candidate_map = {
            str(df.columns[i]): [str(x) for x in df.columns[candidate_mask[i]]]
            for i in range(n_features)
            if np.any(candidate_mask[i])
        }

        # ---------- Step 2: through-origin regression ----------
        X_raw = df.to_numpy(dtype=float)

        missing = missing_func(X_raw)
        missing = np.asarray(missing, dtype=bool)

        if missing.shape != X_raw.shape:
            raise ValueError(
                "missing_func must return a boolean array with the same shape as input data."
            )

        missing = missing | np.isnan(X_raw) | np.isinf(X_raw)

        if intensity_threshold > 0:
            missing = missing | (X_raw <= intensity_threshold)

        valid = ~missing

        X = X_raw.copy()
        X[missing] = 0.0

        G = X.T @ X
        X2 = X ** 2
        V = valid.astype(float)

        ss_x_pair = X2.T @ V
        ss_y_pair = V.T @ X2

        with np.errstate(divide="ignore", invalid="ignore"):
            A = G / ss_x_pair
            R = (G ** 2) / (ss_x_pair * ss_y_pair)

        A[~np.isfinite(A)] = np.nan
        R[~np.isfinite(R)] = np.nan

        A_df = pd.DataFrame(A, index=df.columns, columns=df.columns)
        R_df = pd.DataFrame(R, index=df.columns, columns=df.columns)

        # ---------- Step 3: R^2 filtering ----------
        r2_mask = candidate_mask & (R >= r_square_threshold)

        # ---------- Step 4: isotope upper-bound filtering ----------
        nC_max = np.floor(mz / 12.0).astype(int)
        q = carbon13_abundance / (1.0 - carbon13_abundance)

        ratio_limit = np.full_like(A, np.nan, dtype=float)

        for k in range(1, max_isotope_order + 1):
            limits_k = np.zeros(n_features, dtype=float)

            for i in range(n_features):
                if nC_max[i] >= k:
                    limits_k[i] = math.comb(int(nC_max[i]), k) * (q ** k)
                else:
                    limits_k[i] = 0.0

            mask_k = isotope_order == k
            ratio_limit[mask_k] = np.broadcast_to(
                limits_k[:, None],
                ratio_limit.shape
            )[mask_k]

        ratio_limit = ratio_limit * safety_factor
        ratio_mask = A <= ratio_limit

        final_mask_raw = r2_mask & ratio_mask

        # ---------- Greedy low-m/z assignment ----------
        final_mask = np.zeros_like(final_mask_raw, dtype=bool)
        removed_indices = set()

        for i in np.argsort(mz):
            if i in removed_indices:
                continue

            js = np.where(final_mask_raw[i])[0]
            js = sorted(js, key=lambda j: (isotope_order[i, j], mz[j]))

            for j in js:
                if j not in removed_indices:
                    final_mask[i, j] = True
                    removed_indices.add(j)

        final_rows, final_cols = np.where(final_mask)

        final_table = pd.DataFrame({
            "parent_index": final_rows,
            "isotope_index": final_cols,
            "parent_feature": df.columns[final_rows],
            "isotope_feature": df.columns[final_cols],
            "parent_mz": mz[final_rows],
            "isotope_mz": mz[final_cols],
            "isotope_order": isotope_order[final_rows, final_cols],
            "ppm_error": ppm_error[final_rows, final_cols],
            "slope_A": A[final_rows, final_cols],
            "r_square": R[final_rows, final_cols],
            "max_allowed_ratio": ratio_limit[final_rows, final_cols],
        })

        isotope_features = df.columns[sorted(removed_indices)].tolist()
        parent_features = df.columns[sorted(set(final_rows))].tolist()

        # ---------- write isotope distribution into feature_meta ----------
        fm = self.feature_meta.copy()

        fm["deisotope_role"] = "unique"
        fm.loc[parent_features, "deisotope_role"] = "parent"
        fm.loc[isotope_features, "deisotope_role"] = "isotope"

        fm["isotope_parent"] = pd.NA
        fm["isotope_order"] = pd.NA
        fm["isotope_slope_A"] = np.nan
        fm["isotope_r_square"] = np.nan
        fm["isotope_ppm_error"] = np.nan

        fm["isotope_children"] = "[]"
        fm["n_isotope_children"] = 0

        for k in range(1, max_isotope_order + 1):
            fm[f"M{k}_mz"] = np.nan
            fm[f"M{k}_feature"] = pd.NA
            fm[f"M{k}_slope_A"] = np.nan
            fm[f"M{k}_r_square"] = np.nan
            fm[f"M{k}_ppm_error"] = np.nan
            fm[f"M{k}_max_allowed_ratio"] = np.nan

        for _, row in final_table.iterrows():
            parent = row["parent_feature"]
            iso = row["isotope_feature"]
            k = int(row["isotope_order"])

            fm.loc[iso, "isotope_parent"] = parent
            fm.loc[iso, "isotope_order"] = k
            fm.loc[iso, "isotope_slope_A"] = row["slope_A"]
            fm.loc[iso, "isotope_r_square"] = row["r_square"]
            fm.loc[iso, "isotope_ppm_error"] = row["ppm_error"]

            child_info = {
                "feature": str(iso),
                "mz": float(row["isotope_mz"]),
                "order": k,
                "slope_A": float(row["slope_A"]),
                "r_square": float(row["r_square"]),
                "ppm_error": float(row["ppm_error"]),
                "max_allowed_ratio": float(row["max_allowed_ratio"]),
            }

            old_children = json.loads(fm.loc[parent, "isotope_children"])
            old_children.append(child_info)
            fm.loc[parent, "isotope_children"] = json.dumps(old_children, ensure_ascii=False)
            fm.loc[parent, "n_isotope_children"] = len(old_children)

            fm.loc[parent, f"M{k}_mz"] = row["isotope_mz"]
            fm.loc[parent, f"M{k}_feature"] = iso
            fm.loc[parent, f"M{k}_slope_A"] = row["slope_A"]
            fm.loc[parent, f"M{k}_r_square"] = row["r_square"]
            fm.loc[parent, f"M{k}_ppm_error"] = row["ppm_error"]
            fm.loc[parent, f"M{k}_max_allowed_ratio"] = row["max_allowed_ratio"]

        # ---------- Merge isotope intensities ----------
        processed_data = df.copy()

        if merge_mode == "sum":
            for parent_idx, isotope_idx in zip(final_rows, final_cols):
                parent_col = df.columns[parent_idx]
                isotope_col = df.columns[isotope_idx]

                processed_data[parent_col] = (
                    processed_data[parent_col].fillna(0)
                    + df[isotope_col].fillna(0)
                )

        if remove:
            processed_data = processed_data.drop(columns=isotope_features)
            fm = fm.loc[processed_data.columns].copy()

        result = {
            "candidate_map": candidate_map,
            "candidate_table": candidate_table,
            "A": A_df,
            "R": R_df,
            "ratio_limit": pd.DataFrame(
                ratio_limit,
                index=df.columns,
                columns=df.columns
            ),
            "final_table": final_table,
            "isotope_features": isotope_features,
            "parent_features": parent_features,
            "processed_data": processed_data,
            "feature_meta": fm,
            "params": {
                "isotope_diff": isotope_diff,
                "ppm_tol": ppm_tol,
                "max_isotope_order": max_isotope_order,
                "r_square_threshold": r_square_threshold,
                "carbon13_abundance": carbon13_abundance,
                "intensity_threshold": intensity_threshold,
                "safety_factor": safety_factor,
                "merge_mode": merge_mode,
                "remove": remove,
            }
        }

        if inplace:
            self.deisotope_result = result
            self.data = processed_data
            self.feature_meta = fm

            if not hasattr(self, "file_meta") or self.file_meta is None:
                self.file_meta = {}

            self.file_meta["deisotope"] = {
                "params": result["params"],
                "n_candidate_pairs": int(len(candidate_table)),
                "n_final_isotope_pairs": int(len(final_table)),
                "n_removed_features": int(len(isotope_features)),
                "merge_mode": merge_mode,
            }

            return self

        return result
    
    def get_annotation(self, sdf_path:str, ppm_tol:float, search_mode: Literal["pos", "neg", "both"] = "pos",
                 adducts_pos:dict = DEFAULT_ADDUCTS_POS, 
                 adducts_neg:dict = DEFAULT_ADDUCTS_NEG, **kwargs):
        searcher = SDFMzSearcher(sdf_path=sdf_path, adducts_pos=adducts_pos, adducts_neg=adducts_neg)
        res = searcher.search(mz=self.data.columns.astype(float), ppm_tol=ppm_tol, mode=search_mode, **kwargs)
        return res