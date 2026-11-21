"""Symmetric modulation evidence; this module does not diagnose a fault."""
from itertools import combinations
import math

import numpy as np
from scipy import signal

from . import matcher


SETTINGS = {"carrier_harmonics": 3, "sideband_orders": [1, 2], "peak_half_width_bins": 2.5,
            "symmetry_tolerance_bins": 1, "shaft_exclusion_bins": 2,
            "background_minimum_hz": 5, "cross_band_carrier_tolerance_bins": 2, "minimum_bands": 2}


def symmetric_groups(frequencies_hz, amplitudes, *, family, fundamental_hz, shaft_hz,
                     tolerance_hz_fn, analysis_end_hz, min_relative_amplitude=.1,
                     min_robust_z=4.5, background_radius_bins=12, shaft_speed_relative_uncertainty=.005):
    f, a, df = matcher._spectrum_arrays(frequencies_hz, amplitudes)
    base, shaft = float(fundamental_hz), float(shaft_hz)
    if not all(math.isfinite(v) and v > 0 for v in (base, shaft, analysis_end_hz)):
        raise ValueError("Modulation requires positive finite frequencies and analysis range")
    if min_relative_amplitude < 0 or min_robust_z <= 0 or background_radius_bins < 1:
        raise ValueError("Invalid modulation peak thresholds")
    if not math.isfinite(shaft_speed_relative_uncertainty) or not 0 <= shaft_speed_relative_uncertainty <= .1:
        raise ValueError("Invalid shaft speed uncertainty")
    end = min(len(f)-2, int(np.searchsorted(f, min(analysis_end_hz, base*3+2*shaft+20), side="right"))-1)
    if end < 2:
        return []
    peaks = signal.find_peaks(a[1:end+1])[0]+1
    peak_indices = set(peaks.tolist())
    half = SETTINGS["peak_half_width_bins"]*df

    def measure(target, exact=None):
        if target-half < f[1] or target+half > f[end]:
            return None
        low = int(np.searchsorted(f, target-half))
        high = int(np.searchsorted(f, target+half, side="right"))
        index = int(exact if exact is not None else low+np.argmax(a[low:high]))
        median, sigma = matcher._local_background_stats(a, index, math.ceil(half/df), 1, end,
            max(background_radius_bins, int(SETTINGS["background_minimum_hz"]/df)))
        z = float((a[index]-median)/max(sigma, np.finfo(float).eps))
        if index not in peak_indices or a[index] <= 0 or z < min_robust_z:
            return None
        return {"observed_hz": float(f[index]), "amplitude": float(a[index]), "local_robust_z": z}

    groups = []
    for harmonic in range(1, SETTINGS["carrier_harmonics"]+1):
        tolerance = float(tolerance_hz_fn(harmonic))
        if not math.isfinite(tolerance) or tolerance < 0:
            raise ValueError("Carrier frequency tolerance must be finite and nonnegative")
        centers = peaks[np.abs(f[peaks]-base*harmonic) <= tolerance]
        for index in centers:
            center = measure(f[index], int(index))
            if center is None:
                continue
            for order in SETTINGS["sideband_orders"]:
                lower = measure(f[index]-order*shaft)
                upper = measure(f[index]+order*shaft)
                if lower is None or upper is None:
                    continue
                triple = [lower, center, upper]
                maximum = max(p["amplitude"] for p in triple)
                if min(p["amplitude"] for p in triple) < min_relative_amplitude*maximum:
                    continue
                symmetry = abs((lower["observed_hz"]+upper["observed_hz"])/2-center["observed_hz"])
                if symmetry > SETTINGS["symmetry_tolerance_bins"]*df:
                    continue
                distance = min(abs(p["observed_hz"]-round(p["observed_hz"]/shaft)*shaft) for p in triple)
                margin = min(abs(p["observed_hz"]-round(p["observed_hz"]/shaft)*shaft)
                    -max(SETTINGS["shaft_exclusion_bins"]*df, abs(p["observed_hz"])*shaft_speed_relative_uncertainty)
                    for p in triple)
                groups.append({"family": family, "carrier_harmonic": harmonic, "sideband_order": order,
                    "carrier_hz": center["observed_hz"], "peaks": triple, "symmetry_error_hz": symmetry,
                    "minimum_distance_to_nominal_shaft_grid_hz": distance,
                    "minimum_shaft_separation_margin_hz": margin,
                    "separated_from_nominal_shaft_grid": margin > 0})
    return groups


def cross_band_summary(groups, family, frequency_grid_hz):
    eligible = [g for g in groups if g["family"] == family and g["separated_from_nominal_shaft_grid"]]
    best = []
    for seed in eligible:
        cluster = [g for g in eligible if g["carrier_harmonic"] == seed["carrier_harmonic"]
                   and g.get("envelope_mode") == seed.get("envelope_mode")
                   and g["sideband_order"] == seed["sideband_order"]
                   and abs(g["carrier_hz"]-seed["carrier_hz"]) <= SETTINGS["cross_band_carrier_tolerance_bins"]*frequency_grid_hz]
        pairs = [(a, b) for a, b in combinations(cluster, 2)
                 if a["band_hz"][1] <= b["band_hz"][0] or b["band_hz"][1] <= a["band_hz"][0]]
        if pairs:
            representative = max(pairs, key=lambda pair: min(p["local_robust_z"] for g in pair for p in g["peaks"]))
            if not best or min(p["local_robust_z"] for g in representative for p in g["peaks"]) > min(p["local_robust_z"] for g in best for p in g["peaks"]):
                best = list(representative)
    return {"supported": bool(best), "representative_groups": best,
            "assessment": ("两个不重叠频带重复观测到对称边带；支持调制结构，不能单独定位故障或排除其他旋转来源"
                           if best else "未达到跨频带对称调制支持，不能据此判正常或异常"),
            "shaft_reference": "nominal rpm only; no measured phase or tachometer"}
