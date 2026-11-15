"""Read acquisition settings without loading spectral intensity arrays."""
from pathlib import Path
import re
import xml.etree.ElementTree as ET

import numpy as np


def resolve_ms_files(inputs):
    paths = [inputs] if isinstance(inputs, (str, Path)) else inputs
    files = []
    for value in paths:
        path = Path(value).expanduser()
        candidates = sorted(path.iterdir()) if path.is_dir() and path.suffix.lower() != ".raw" else [path]
        files.extend(p.resolve() for p in candidates
                     if p.suffix.lower() in {".mzml", ".mzxml", ".raw"} and p.exists())
    return list(dict.fromkeys(files))


def _xml_settings(path, max_scans=32):
    windows, resolutions, groups = [], [], {}
    count = 0
    with open(path, "rb") as handle:
        for _, element in ET.iterparse(handle, events=("end",)):
            tag = element.tag.rsplit("}", 1)[-1]
            if tag == "referenceableParamGroup":
                groups[element.get("id")] = [dict(item.attrib) for item in element]
            if tag not in {"spectrum", "scan"}:
                continue
            if tag == "scan" and "msLevel" not in element.attrib:
                continue
            params = [item.attrib for item in element.iter()
                      if item.tag.rsplit("}", 1)[-1] in {"cvParam", "userParam"}]
            for item in element.iter():
                if item.tag.endswith("referenceableParamGroupRef"):
                    params.extend(groups.get(item.get("ref"), []))
            level = element.get("msLevel") or next((p.get("value") for p in params
                                                    if p.get("accession") == "MS:1000511"), None)
            if str(level) == "1":
                count += 1
                lower = [float(p["value"]) for p in params if p.get("accession") == "MS:1000501"]
                upper = [float(p["value"]) for p in params if p.get("accession") == "MS:1000500"]
                if lower and upper:
                    windows.append((min(lower), max(upper)))
                elif tag == "scan" and element.get("lowMz") and element.get("highMz"):
                    windows.append((float(element.get("lowMz")), float(element.get("highMz"))))
                else:
                    filters = [p.get("value", "") for p in params if p.get("accession") == "MS:1000512"]
                    filters.append(element.get("filterLine", ""))
                    for text in filters:
                        match = re.search(r"\[\s*([\d.]+)\s*-\s*([\d.]+)\s*\]", text)
                        if match:
                            windows.append(tuple(map(float, match.groups())))
                for p in params:
                    name = p.get("name", "").lower()
                    if p.get("value") and (p.get("accession") == "MS:1000800"
                                           or name in {"orbitrap resolution", "ft resolution", "resolution"}):
                        resolutions.append(float(p["value"]))
                if element.get("resolution"):
                    resolutions.append(float(element.get("resolution")))
            element.clear()
            if count >= max_scans:
                break
    return windows, resolutions, count


def read_acquisition_settings(paths):
    """Inspect up to 32 MS1 scans per file, using explicit acquisition metadata."""
    rows = []
    for path in resolve_ms_files(paths):
        row = {"file": str(path)}
        try:
            if path.suffix.lower() == ".raw":
                from .thermo_raw import read_raw_acquisition_settings
                windows, resolutions, count = read_raw_acquisition_settings(path)
            else:
                windows, resolutions, count = _xml_settings(path)
            windows = [(low, high) for low, high in windows if 0 < low < high]
            resolutions = [value for value in resolutions if np.isfinite(value) and value > 0]
            row.update(scans=count, windows=windows, resolutions=sorted(set(resolutions)))
        except Exception as exc:
            row["error"] = str(exc)
        rows.append(row)
    windows = [window for row in rows for window in row.get("windows", [])]
    resolutions = [value for row in rows for value in row.get("resolutions", [])]
    result = {"files": rows}
    if windows:
        result.update(mz_min=min(w[0] for w in windows), mz_max=max(w[1] for w in windows))
    if resolutions:
        result["resolution"] = min(resolutions)
    return result
