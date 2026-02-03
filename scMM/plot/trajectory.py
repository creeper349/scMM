import json
import subprocess
import pandas as pd
import numpy as np
import tempfile
import os
import warnings
import gzip
import logging

from ..file.data import CyESIData
from ._traj import PseudoTrajectory
from ._trajplot import *
from ._sc import *

pkg_dir = os.path.dirname(__file__)
R_SCRIPT_PATH = os.path.join(pkg_dir, "R", "traj.R")

if not os.path.exists(R_SCRIPT_PATH):
    raise FileNotFoundError(f"R script not found: {R_SCRIPT_PATH}")

def _traj_r(data_obj:CyESIData, method:str = "dpt", **params):
    """
    Run a R-subprocess to construct pseudotime trajectory of a single-cell metabolite dataset.
    
    :param data_obj: Processed CyESI data
    :type data_obj: CyESIData
    :param method: Pseudotime method for construction, e.g. "slingshot", "dpt", "tscan", 
    could be extended by adding more R modules
    :type method: str
    :param params: Parameters for pseudotime function
    """
    X = data_obj.data.copy()
    X = X.astype(np.float64)
    if X.index is None or X.index.dtype == int:
        X.index = [f"cell{i}" for i in range(X.shape[0])]
    if X.columns is None or X.columns.dtype == int:
        X.columns = [f"feat{i}" for i in range(X.shape[1])]
        
    obs = pd.DataFrame(index=X.index)
    var = pd.DataFrame(index=X.columns)

    tmp_X = tempfile.NamedTemporaryFile(suffix=".csv", dir=".", delete=False).name
    tmp_obs = tempfile.NamedTemporaryFile(suffix=".csv", dir=".", delete=False).name
    tmp_var = tempfile.NamedTemporaryFile(suffix=".csv", dir=".", delete=False).name
    tmp_output = tempfile.NamedTemporaryFile(suffix=".json.gz", dir=".", delete=False).name
    tmp_params = tempfile.NamedTemporaryFile(suffix=".json", dir=".", delete=False).name

    X.to_csv(tmp_X, float_format="%.8g")
    obs.to_csv(tmp_obs)
    var.to_csv(tmp_var)

    with open(tmp_params, "w") as f:
        json.dump(params or {}, f)

    try:
        subprocess.run([
            "Rscript", R_SCRIPT_PATH,
            tmp_X,
            tmp_obs,
            tmp_var,
            tmp_params,
            method,
            tmp_output,
            pkg_dir
        ], check=True, cwd=pkg_dir)

        with gzip.open(tmp_output, "rt", encoding="utf-8") as f:
            result_raw = json.load(f)
        coords = pd.DataFrame(result_raw["coordinates"], index=X.index)
        pseudotime = pd.Series(result_raw["pseudotime"], index=X.index)
        branch = pd.Series(result_raw["branch"], index=X.index)

        result = {"coordinates": coords, "pseudotime": pseudotime, "branch": branch}

    except Exception as e:
        warnings.warn(f"Failed to run R script: {e}")
        result = None
        
    finally:
        for f in [tmp_X, tmp_obs, tmp_var, tmp_output, tmp_params]:
            if os.path.exists(f):
                try:
                    os.remove(f)
                except Exception:
                    pass

    return result

def _traj(data_obj:CyESIData, fig_path_dir:str = None, embed_method="umap", start_idx:int = 0, n_components_hd=15, n_neighbors=15, **params):
    traj = PseudoTrajectory(data_obj.data.values)
    traj.run(embed_method = embed_method, 
                      start_idx = start_idx,
                      n_components_hd = n_components_hd,
                      n_neighbors = n_neighbors,
                      **params)
    if (fig_path_dir is not None) and os.path.isdir(fig_path_dir):
        plot_pseudotime(traj, 
                        figpath = os.path.join(fig_path_dir, f"{data_obj.get_name()}_pseudotime.svg"), 
                        point_size = params.get("s", 10),
                        cmap = params.get("cmap", "viridis"))
        plot_branches(traj, 
                      figpath = os.path.join(fig_path_dir, f"{data_obj.get_name()}_branches.svg"), 
                      point_size = params.get("s", 10))
    return traj

def _traj_scanpy(data_obj:CyESIData, fig_path_dir:str, method:str = "palantir", start_idx:int = 0):
    engine = (
        PseudotimeEngine(data_obj.data, fig_path_dir=fig_path_dir)
        .set_root_by_index(start_idx)
        .build_graph(n_neighbors=20)
    )
    engine.plot_umap("class", data_obj.get_time())
    if method == "dpt":
        engine.run_dpt(root_key="is_root", root_value=True)
    elif method == "palantir":
        engine.run_palantir(root_key="is_root", root_value=True)
    engine.run_paga()
    engine.run_diffmap()
    engine.plot_trajectory()
    engine.plot_trend_clusters()

def run_traj(data_obj:CyESIData, method:str = None, backend:str = "scanpy", **kwargs):
    logging.info(f"Run trajectory analysis on {data_obj.get_name()}, backend:{backend}")
    if backend == "scmm":
        return _traj(data_obj=data_obj, 
                            embed_method = method, 
                            fig_path_dir = kwargs.get("fig_path_dir", None),
                            start_idx = kwargs.get("start_idx", 0),
                            n_components_hd = kwargs.get("n_components_hd", 15),
                            n_neighbors = kwargs.get("n_neighbors", 15),
                            **kwargs)
    elif backend == "r":
        return _traj_r(data_obj, method, **kwargs)
    elif backend == "scanpy":
        return _traj_scanpy(data_obj, start_idx = kwargs.get("start_idx", 0), 
                            fig_path_dir = kwargs.get("fig_path_dir", None))