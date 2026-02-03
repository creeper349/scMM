from .file.batch import run_batch
from .util.decorator import timer
import numpy as np

@timer
def run():
    run_batch("/home/zby/scMM/data/algea-0117",
            "/home/zby/scMM/file/0117-processed",
            data_kwargs={
                "ref_mz": 734.5929,
                "dtype": np.float32
            },
            preprocess_kwargs={"subtract_baseline": False},
            method='multithreading')
    
if __name__=="__main__":
    run()