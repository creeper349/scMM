# scMM

`scMM` is a Python package for processing and analysing CyESI single-cell mass
spectrometry data. It provides spectrum loading and alignment, cell peak
detection, normalisation, isotope removal, annotation, and downstream plotting.

## Development environment

Dependencies are managed with Conda so that compiled scientific packages such
as OpenMS are installed consistently:

```bash
conda env create -f environment.yml
conda activate scmm-dev
python -m pip install --no-deps -e .
```

After `environment.yml` changes, update the existing environment with:

```bash
conda env update -f environment.yml --prune
```

## Quality checks

Run the complete local verification suite from the repository root:

```bash
ruff format --check .
ruff check .
pytest -W error
python -m build
```

## Command-line processing

Process either one mzML/mzXML file or every supported file in a directory:

```bash
scmm-process input.mzML results --ref-mz 734.5929
scmm-process raw-data/ results --ref-mz 734.5929 --jobs 4
```

Use `scmm-process --help` for all preprocessing options.

## Python API

```python
from scMM.file.data import CyESIData

data = CyESIData.load_from_file(
    "sample.mzML",
    ref_mz=734.5929,
    cell_snr=5.0,
    peak_snr=3.0,
)
data.normalize("total")
result_dir = data.save("results")
```

Processed datasets can be reopened with `CyESIData(result_dir)`. Saving writes
both pickle files for faithful and efficient reloads and CSV files for
inspection/interchange.
