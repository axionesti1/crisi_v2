from __future__ import annotations
from typing import Dict

# CRISI internal (old NUTS2 codes) -> Region name
NUTS2_REGIONS: Dict[str, str] = {
    "EL11": "Anatoliki Makedonia, Thraki",
    "EL12": "Kentriki Makedonia",
    "EL13": "Dytiki Makedonia",
    "EL14": "Thessalia",
    "EL21": "Ipeiros",
    "EL22": "Ionia Nisia",
    "EL23": "Dytiki Ellada",
    "EL24": "Sterea Ellada",
    "EL25": "Peloponnisos",
    "EL30": "Attiki",
    "EL41": "Voreio Aigaio",
    "EL42": "Notio Aigaio",
    "EL43": "Kriti",
}

# Crosswalk: internal (old) -> display (new)
CRISI_TO_GEO: Dict[str, str] = {
    "EL11": "EL51",
    "EL12": "EL52",
    "EL13": "EL53",
    "EL14": "EL61",
    "EL21": "EL54",
    "EL22": "EL62",
    "EL23": "EL63",
    "EL24": "EL64",
    "EL25": "EL65",
    "EL30": "EL30",
    "EL41": "EL41",
    "EL42": "EL42",
    "EL43": "EL43",
}

# Reverse crosswalk
GEO_TO_CRISI: Dict[str, str] = {v: k for k, v in CRISI_TO_GEO.items()}


def ui_region_label(internal_id: str) -> str:
    """
    UI label uses ONLY the NEW code + name.
    The selectbox still returns the internal_id as the selected value.
    """
    name = NUTS2_REGIONS.get(internal_id, internal_id)
    new_id = CRISI_TO_GEO.get(internal_id, internal_id)
    return f"{new_id} — {name}"


