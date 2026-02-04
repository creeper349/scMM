import pandas as pd
from anndata import AnnData
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