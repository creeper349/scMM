"""LIPID MAPS species-level names for direct-infusion reports."""
import re
import threading
from functools import lru_cache

from pygoslin.domain.LipidExceptions import LipidException
from pygoslin.domain.LipidLevel import LipidLevel
from pygoslin.parser.Parser import LipidParser

_local = threading.local()
_REGULAR_CLASSES = set("PC PE PS PI PG PA PGP PIP PIP2 PIP3 LPC LPE LPS LPI LPG LPA CL MLCL DLCL BMP LBPA MG DG TG MGDG DGDG SQDG SQMG DGTS DGTA LDGTS LDGTA Cer CerP CerPE SM SPB SPBP HexCer Hex2Cer SE".split())


def _coarse_headgroup(name):
    # Hexose stereochemistry and glycosidic linkages are outside this report level.
    name = re.sub(r"\b(?:Glc|Gal|Man)(?=Cer\b)", "Hex", name)
    name = re.sub(r"\bLacCer\b", "Hex2Cer", name)
    match = re.match(r"(.+?)Cer(?= )", name)
    if match:
        prefix = match[1].replace("-", "")
        tokens = list(re.finditer(r"(NeuAc|NeuGc|HexNAc|HexA|Hex|Fuc|S)(?:\((\d+)\)|(\d+))?", prefix))
        if "".join(t[0] for t in tokens) == prefix:
            counts = {}
            for token in tokens:
                counts[token[1]] = counts.get(token[1], 0) + int(token[2] or token[3] or 1)
            head = "".join(k + (str(counts[k]) if counts[k] > 1 else "") for k in sorted(counts, key=lambda k: (k != "S", k)))
            name = head + name[match.end(1):]
    return name


@lru_cache(maxsize=65536)
def lipid_species(name):
    """Return a species name, or an empty string when the grammar cannot resolve it."""
    name = re.sub(r"\s*\(LM[A-Z]{2}\d+\)\s*$", "", str(name)).strip()
    name = re.sub(r"\b([dmt])\s+(?=\d+:)", r"\1", name)
    name = re.sub(r"^MGD(?=\s|\()", "MGDG", name)
    name = re.sub(r"\b(PIP[23]?)\([^:)]*\)\s*\(", r"\1(", name)
    alkyne = re.fullmatch(r"(FA \d+:)(\d+);((?:\d*Y)(?:,\d*Y)*|\(Y\)\d+)", name)
    if alkyne:
        count = int(alkyne[3][3:]) if alkyne[3].startswith("(Y)") else alkyne[3].count("Y")
        name = alkyne[1] + str(int(alkyne[2]) + 2 * count)
    if not name:
        return ""
    if re.fullmatch(r"SE \d+:\d+(?:;O\d*)?", name):
        return name
    if not hasattr(_local, "parser"):
        _local.parser = LipidParser()
    try:
        parsed = _local.parser.parse(name)
        species = parsed.lipid.get_lipid_string(LipidLevel.SPECIES)
        isotopes = parsed.adduct.get_heavy_isotope_string() if parsed.adduct else ""
    except LipidException:
        betaine_name = re.sub(r"\(\d+[ZE]?(?:,\d+[ZE]?)*\)", "", name)
        betaine = re.fullmatch(r"(L?DGT[SA])\s*\(?((?:\d+:\d+)(?:[/_]\d+:\d+)*)\)?", betaine_name)
        if betaine:
            chains = [tuple(map(int, p.split(":"))) for p in re.split(r"[/_]", betaine[2])]
            return f"{betaine[1]} {sum(c for c, d in chains)}:{sum(d for c, d in chains)}"
        return ""
    # Goslin 2.2.5 keeps the sterol/acyl split even at SPECIES level.
    sterol = re.fullmatch(r"SE (\d+):(\d+)(;O\d*)?[/_](\d+):(\d+)(;O\d*)?", species)
    if sterol:
        c1, d1, o1, c2, d2, o2 = sterol.groups()
        oxygen = sum(int(o[2:] or 1) for o in (o1, o2) if o)
        species = f"SE {int(c1)+int(c2)}:{int(d1)+int(d2)}" + (f";O{oxygen if oxygen > 1 else ''}" if oxygen else "")
    return _coarse_headgroup(species) + (f"[M{isotopes}]" if isotopes else "")


def normalize_lipid_name(name):
    return _regular_species(str(name)) or str(name).strip()


def _regular_species(name):
    species = lipid_species(name)
    head = species.split(" ", 1)[0]
    if head in _REGULAR_CLASSES or re.fullmatch(r"ST \d+:\d+;O(?:\[M.*\])?", species):
        return species
    return ""


def report_lipid_name(name, abbreviation="", category="", formula=""):
    """Simplify regular lipid families and retain every other original compound name."""
    for value in (abbreviation, name):
        species = _regular_species(str(value)) if value else ""
        if species:
            return species, "species"
    return str(name).strip(), "unchanged"
