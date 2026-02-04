from .file.data import CyESIData
from .util.decorator import timer
from .plot.msplot import plot_hook
from .file.batch import align_batch
import numpy as np

import logging
logging.basicConfig(level=logging.INFO)

@timer
def run():
    data = CyESIData("/home/zby/scMM/file/raw/2.mzML", ref_mz=734.5929, dtype=np.float32)
    data.preprocess(debug_hook=plot_hook).impute('knn')
    data.save("/home/zby/scMM/file/result/")
    
@timer
def concat():
    align_batch("/home/zby/scMM/file/0117-processed", "/home/zby/scMM/file/concat", base = "1709-add")
    
if __name__=="__main__":
    """
    from .plot.embedding import dimension_reduction
    import matplotlib.pyplot as plt
    data = CyESIData.load_from_processed("/home/zby/scMM/file/concat/1709-add", dtype=np.float32).normalize("quantile")
    dimension_reduction(data, method = "umap", color = data[832.3437], reduce_kwargs={"n_neighbors": 80}, plot_kwargs = {"s": 2, "palette": "tab20"})
    plt.savefig(f"umap_label_quantile_832.svg")
    """
    """
    from .plot.trajectory import PseudotimeEngine
    from .plot.mpl_style import *
    data = CyESIData.load_from_processed("/home/zby/src/files/concat/1709-add", dtype=np.float32).normalize("quantile")
    PseudotimeEngine.from_CyESIData(data, "/home/zby/src/files/result/fig")\
        .set_root_by_index(4158)\
        .normalize("quantile").use_layer("norm_quantile")\
        .decomposition("umap", color_key = "time", reduce_kwargs={"n_neighbors": 50})\
        .build_graph().run_palantir(root_key="is_root", root_value = True).diffmap()\
        .plot_trajectory().save_adata("/home/zby/src/files/result/h2o2.hdf5")"""
    #run_traj(data, fig_path_dir = "/home/zby/scMM/file/result/fig", start_idx = 4158) #4443
    from .plot.trajectory import PseudotimeEngine
    from .plot.mpl_style import *
    engine = PseudotimeEngine.from_adata("/home/zby/src/files/result/h2o2.hdf5", "/home/zby/src/files/result/fig")\
        .plot_trend_clusters("3034").plot_trend_clusters("13489")\
        .plot_trajectory("3034").plot_trajectory("13489").plot_trends(["734.5939", "767.47064", "732.5773"], heatmap=True)
    print(engine.adata.var)