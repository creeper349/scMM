from scMM.file.data import CyESIData
import logging
import numpy as np
logging.basicConfig(level=logging.INFO)

#data = CyESIData.load_from_filelist("/home/zby/scMM/data/3d-models/test", ref_mz = 734.5929, prominence_ratio = 0.01)
data = CyESIData.load_from_filelist("/home/zby/scMM/data/3d-models/20260329-yz-0mM", ref_mz = 734.5929,
                    ppm_tol = 10, cell_snr = 5.0, peak_snr = 1.0, ms_peak_snr_threshold = 10.0, n_jobs=-1, dtype=np.float32)
data.save("/home/zby/scMM/data/algea_results/raw/0mM-1")