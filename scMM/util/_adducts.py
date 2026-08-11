"""Adduct definitions and neutral/ion mass conversion."""

from __future__ import annotations

PROTON = 1.007276466812
NA = 22.989218
K = 38.963158
NH4 = 18.033823
CL = 34.969402
FA = 44.998201


DEFAULT_ADDUCTS_POS = {
    "[M+H]+": {"mass_shift": PROTON, "charge": 1},
    "[M+Na]+": {"mass_shift": NA, "charge": 1},
    "[M+K]+": {"mass_shift": K, "charge": 1},
    "[M+NH4]+": {"mass_shift": NH4, "charge": 1},
}

DEFAULT_ADDUCTS_NEG = {
    "[M-H]-": {"mass_shift": -PROTON, "charge": 1},
    "[M+Cl]-": {"mass_shift": CL, "charge": 1},
    "[M+FA-H]-": {"mass_shift": FA - PROTON, "charge": 1},
}


def theoretical_mz(neutral_mass: float, adduct_info: dict) -> float:
    """Convert neutral mass to theoretical ion m/z for one adduct."""
    charge = adduct_info["charge"]
    if charge == 0:
        raise ValueError("Adduct charge cannot be zero")
    return (neutral_mass + adduct_info["mass_shift"]) / abs(charge)


def neutral_mass_from_mz(mz: float, adduct_info: dict) -> float:
    """Convert observed ion m/z back to neutral mass for one adduct."""
    charge = adduct_info["charge"]
    if charge == 0:
        raise ValueError("Adduct charge cannot be zero")
    return mz * abs(charge) - adduct_info["mass_shift"]
