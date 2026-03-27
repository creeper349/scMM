from .file.batch import run_batch
from .util.decorator import timer
import numpy as np

@timer
def run():
    run_batch("/home/zby/scMM/data/3d-models/20260323-yz-30mM",
            "/home/zby/scMM/data/algea-0324",
            data_kwargs={
                "ref_mz": 734.5929,
                "dtype": np.float32
            },
            preprocess_kwargs={"subtract_baseline": True},
            method='multithreading')
    
if __name__=="__main__":
    run()