"nohup python -m scMM.debug > output.log 2>&1 &"

from .file.data import CyESIData
import logging
logging.basicConfig(level = logging.INFO)

data = CyESIData.load_from_filelist("/home/zby/scMM/data/3d-models/20260329-yz-0mM", ref_mz = 734.5929,
                                    cell_snr = 5.0, peak_snr = 2.0)
data.save("/home/zby/scMM/data/algea_results/0mM")