"""Select envelope bands from the current vibration record."""

from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np

from ._band_selection_maps import band_selection_maps

BAND_METHODS = ("fast_kurtogram", "autogram", "infogram")
BAND_METHOD_NAMES = dict(zip(BAND_METHODS, ("快速谱峭度", "Autogram", "Infogram"), strict=True))


def select_envelope_bands(
    signal: np.ndarray,
    fs: float,
    min_band_hz: float,
) -> dict[str, Any]:
    """Rank bands independently within each of the three selectors."""
    selected = []
    maps = band_selection_maps(signal, fs)
    candidates = {}
    for method in BAND_METHODS:
        result = maps["methods"][method]
        ranked = []
        if result["has_signal_energy"]:
            for row in result["rows"]:
                for cell in row["cells"]:
                    low, high = cell["band_hz"]
                    if (min_band_hz <= low < high <= fs / 2
                            and high - low >= min_band_hz and math.isfinite(cell["score"]) and cell["score"] > 0):
                        ranked.append({**cell, "level": row["level"]})
        ranked.sort(key=lambda cell: -cell["score"])
        candidates[method] = ranked[:5]
        method_bands: list[Sequence[float]] = []
        for cell in ranked:
            low, high = cell["band_hz"]
            if any(
                min(high, right) - max(low, left) >= 0.5 * min(high - low, right - left)
                for left, right in method_bands
            ):
                continue
            selected.append({"method": method, **cell})
            method_bands.append(cell["band_hz"])
            if len(method_bands) == 3:
                break
    return {"primary_method": BAND_METHODS[0], "selected": selected, "candidates": candidates,
            "implementation": maps["implementation"], "modwpt": maps["modwpt"]}
