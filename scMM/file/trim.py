"""Select original spectral frames for processing without changing the source."""
from pathlib import Path


def ranges_for_file(path, file_ranges=None, *, rt_range=None, frame_range=None):
    settings = (file_ranges or {}).get(str(Path(path).resolve()), {})
    return {"rt_range": settings.get("rt_range", rt_range),
            "frame_range": settings.get("frame_range", frame_range)}


def frame_extent(path):
    """Read frame count and retention-time bounds without intensity arrays for XML."""
    import xml.etree.ElementTree as ET
    source = Path(path)
    times, count = [], 0
    if source.suffix.lower() == ".raw":
        from .thermo_raw import _load_api
        adapter, device = _load_api()
        raw = adapter.FileFactory(str(source.resolve()))
        try:
            raw.SelectInstrument(device.MS, 1)
            first, last = int(raw.RunHeaderEx.FirstSpectrum), int(raw.RunHeaderEx.LastSpectrum)
            return {"frames": last - first + 1,
                    "rt_min": float(raw.RetentionTimeFromScanNumber(first)) * 60,
                    "rt_max": float(raw.RetentionTimeFromScanNumber(last)) * 60}
        finally:
            raw.Dispose()
    with source.open("rb") as stream:
        for _, element in ET.iterparse(stream, events=("end",)):
            tag = element.tag.rsplit("}", 1)[-1]
            if tag == "cvParam" and element.get("accession") == "MS:1000016":
                value = float(element.get("value"))
                times.append(value * 60 if element.get("unitAccession") == "UO:0000031" else value)
            if tag == "spectrum" or (tag == "scan" and element.get("msLevel")):
                count += 1
                if element.get("retentionTime"):
                    import re
                    match = re.fullmatch(r"PT(?:(\d+(?:\.\d+)?)H)?(?:(\d+(?:\.\d+)?)M)?(?:(\d+(?:\.\d+)?)S)?", element.get("retentionTime"))
                    if match:
                        hours, minutes, seconds = (float(v or 0) for v in match.groups())
                        times.append(hours * 3600 + minutes * 60 + seconds)
                element.clear()
    if not count or not times:
        raise ValueError(f"{source.name} 中未读取到帧数或保留时间。")
    return {"frames": count, "rt_min": min(times), "rt_max": max(times)}


def selected_frames(experiment, *, ms_level=1, rt_range=None, frame_range=None):
    """Yield original zero-based frame IDs and spectra, with inclusive bounds."""
    for name, bounds in [("保留时间", rt_range), ("帧序号", frame_range)]:
        if bounds is not None and (bounds[0] < 0 or bounds[1] < bounds[0]):
            raise ValueError(f"{name}范围应满足 0 ≤ 下限 ≤ 上限。")
    for frame_id, spectrum in enumerate(experiment):
        if (spectrum.getMSLevel() == ms_level
                and (rt_range is None or rt_range[0] <= spectrum.getRT() <= rt_range[1])
                and (frame_range is None or frame_range[0] <= frame_id <= frame_range[1])):
            yield frame_id, spectrum
