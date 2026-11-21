"""Project algorithms as reproducible dataset evidence; no automatic fault verdict."""
from __future__ import annotations

from copy import deepcopy
import math
import re

import numpy as np
from scipy import signal as scipy_signal

from . import diagnosability, geometry, matcher, signal_tools
from .assessment import domain_indicators as indicators
from .config import load_config
from .diagnosis_schemas import FAULT_TO_FAMILY
from .features import welch_psd_features
from .speed_estimation import SETTINGS as SPEED_SETTINGS, estimate_shaft_speed


FAMILIES = ("BPFI", "BPFO", "BSF")
LATTICE_SETTINGS = {"relative_span": 0.05, "refine_steps": 401, "peak_half_bins": 2,
                    "min_orders_for_fit": 3, "visibility_sigma": 2.0, "accept_ratio": 0.6,
                    "modulation_hz": None, "max_sideband_order": 0}
FAULT_NAMES = {"normal": "正常", "inner_race_fault": "内圈故障", "outer_race_fault": "外圈故障",
               "rolling_element_fault": "滚动体故障", "abnormal": "异常"}
HEADINGS = ("【分析条件】", "【整体表现】", "【故障特征证据】", "【综合判断】")


def public_input(record):
    return deepcopy({
        "signal": {key: record["signal"][key] for key in ("channel", "sampling_rate_hz", "segment_samples", "unit") if key in record["signal"]},
        "equipment": record["equipment"], "operating_condition": record["operating_condition"],
    })


def _tolerance(family, fundamental, harmonic, df, matching):
    return matcher.harmonic_tolerance_hz(fundamental, harmonic, df,
        speed_relative_uncertainty=matching["speed_relative_uncertainty"],
        geometry_relative_uncertainty=matching["geometry_relative_uncertainty"],
        slip_relative_uncertainty=matching["slip_relative_uncertainty_by_family"][family],
        resolution_multiplier=matching["tolerance_resolution_multiplier"],
        minimum_tolerance_hz=matching["minimum_tolerance_hz"])


def harmonic_support_pattern(orders, minimum_orders=3):
    """Describe whether supported orders contain the fundamental and a continuous run.

    This is descriptive. ``fault_specific_support`` keeps the existing threshold, so
    sparse combinations stay recorded as support; the pattern tells whether the support
    has the low-order continuity expected from a single bearing fault comb.
    """
    values = sorted({int(order) for order in orders})
    if not values:
        return {"low_order_anchor": False, "longest_consecutive_orders": [],
                "support_pattern": None}
    longest = run = [values[0]]
    for order in values[1:]:
        run = run + [order] if order == run[-1] + 1 else [order]
        if len(run) > len(longest):
            longest = run
    anchored = bool(1 in values and len(longest) >= int(minimum_orders))
    return {"low_order_anchor": 1 in values, "longest_consecutive_orders": longest,
            "support_pattern": "anchored_contiguous" if anchored else "sparse"}


def verify_lattice(result, family, fundamental, df, matching, peaks, observable_orders):
    """Check physical attribution against the same strictly matched spectral peaks."""
    model = result["models"]["lattice"]
    fitted = model.get("fitted_frequency_hz")
    lines = {line["order"]: line for line in model.get("lines", [])
             if line["order"] in observable_orders and line["order"] != model.get("dropped_order")
             and line.get("sideband", 0) == 0}
    visible = {order for order, line in lines.items() if line["visible"]}
    tolerance = min((_tolerance(family, fundamental, order, df, matching)/order for order in visible),
                    default=_tolerance(family, fundamental, 1, df, matching))
    error = abs(fitted-fundamental) if fitted is not None and math.isfinite(fitted) else None
    consistent = error is not None and error <= tolerance + 1e-10
    corresponding = sorted(peak["harmonic"] for peak in peaks
        if peak["qualified"] and peak["harmonic"] in visible
        and abs(peak["observed_frequency_hz"]-lines[peak["harmonic"]]["observed_frequency_hz"]) <= df/2)
    minimum = LATTICE_SETTINGS["min_orders_for_fit"]
    supported = bool(result["best_model"] == "lattice" and not result["fitted_is_interference"]
                     and consistent and len(corresponding) >= minimum)
    pattern = harmonic_support_pattern(corresponding, minimum)
    if error is None:
        assessment = "拟合频率不可用，无法核验故障归属"
    elif not consistent:
        assessment = "拟合间隔超出理论频率允许偏差，不能归属于该故障族"
    elif result["fitted_is_interference"]:
        assessment = "拟合规律与转频倍频通约，缺乏独立的故障定位依据"
    elif result["best_model"] == "shaft":
        assessment = "当前谱峰排列更符合转频解释"
    elif supported and pattern["support_pattern"] == "sparse":
        if pattern["low_order_anchor"]:
            assessment = "同组谱峰满足理论容差、严格峰值筛选及周期拟合，但同组阶次不连续，仅形成稀疏候选支持"
        else:
            assessment = "同组谱峰满足理论容差、严格峰值筛选及周期拟合，但缺少第1阶锚点，仅形成稀疏候选支持"
    elif supported:
        assessment = "同组谱峰满足理论容差、严格峰值筛选及周期拟合，形成故障倍频支持"
    elif result["best_model"] == "lattice":
        assessment = f"仅有候选周期结构；同组严格匹配谐波不足{minimum}阶，不能据此确认故障归属"
    else:
        assessment = "未形成被当前判据接受的周期结构，不直接表示正常或故障"
    return {"fundamental_error_hz": error, "fundamental_tolerance_hz": tolerance,
            "frequency_consistent": consistent, "corresponding_harmonics": corresponding,
            "fault_specific_support": supported, "assessment": assessment, **pattern}


def autogram_settings(frequencies, df, matching):
    """Budget a common carrier span for the existing minimum harmonic fit, without labels."""
    from .autogram_selection import SETTINGS

    settings = dict(SETTINGS)
    required_order = LATTICE_SETTINGS["min_orders_for_fit"]
    required_end = max(frequencies[family]*required_order
        + _tolerance(family, frequencies[family], required_order, df, matching)
        for family in FAMILIES) + LATTICE_SETTINGS["peak_half_bins"]*df
    transition_ratio = min(.2, max(.01, settings["filterbank_transition_ratio"]))
    # Reserve both FFT-mask transition edges; the same conservative span gates downstream views.
    minimum_width = 2*max(required_end + 4*df, required_end/(1-2*transition_ratio)) + 2*df
    settings["configured_minimum_bandwidth_hz"] = settings["minimum_bandwidth_hz"]
    settings["configured_maximum_bandwidth_hz"] = settings["maximum_bandwidth_hz"]
    settings["minimum_bandwidth_hz"] = max(settings["minimum_bandwidth_hz"], minimum_width)
    settings["maximum_bandwidth_hz"] = max(settings["maximum_bandwidth_hz"], 2*settings["minimum_bandwidth_hz"])
    settings["coverage_required_harmonics"] = list(range(1, required_order+1))
    settings["minimum_coverage_analysis_end_hz"] = required_end
    physical_upper = 2*max(frequencies[family] for family in FAMILIES)
    settings["configured_autogram_maximum_cyclic_frequency_hz"] = settings["autogram_maximum_cyclic_frequency_hz"]
    settings["physical_autogram_maximum_cyclic_frequency_hz"] = physical_upper
    settings["autogram_maximum_cyclic_frequency_hz"] = min(settings["autogram_maximum_cyclic_frequency_hz"], physical_upper)
    return settings


def autogram_coverage(band_hz, fs, df, frequencies, matching):
    """Declare conservative coverage; do not equate out-of-range with physical absence."""
    from .autogram_selection import SETTINGS

    width = band_hz[1]-band_hz[0]
    transition = min(max(4*df, SETTINGS["filterbank_transition_ratio"]*width), .25*width)
    end = max(0., min(width/2-transition, fs/2))
    required = list(range(1, LATTICE_SETTINGS["min_orders_for_fit"]+1))
    families = {}
    for family in FAMILIES:
        observable = [h for h in range(1, int(matching["max_harmonics"])+1)
            if frequencies[family]*h+_tolerance(family, frequencies[family], h, df, matching)
            < end-LATTICE_SETTINGS["peak_half_bins"]*df]
        families[family] = {"observable_harmonics": observable,
            "required_orders_covered": all(h in observable for h in required)}
    return {"basis": "conservative_centered_carrier_with_filter_edge_reserve",
            "band_hz": list(band_hz), "filter_edge_reserve_hz": transition, "analysis_end_hz": end,
            "required_harmonics": required, "families": families,
            "normal_assessable": all(v["required_orders_covered"] for v in families.values()),
            "assessment": "居中载波及滤波边缘预留下的保守核验范围；不可观测不等于物理上不存在；覆盖不是故障峰已出现或正常的证明"}


def normal_coverage_issue(evidence):
    for row in evidence.get("extracted_features", []):
        if row.get("name", "").endswith(".selection.coverage") and row["value"].get("normal_assessable") is False:
            return "主解调频带关键故障频率覆盖不足，不能形成正常结论；保留待复核，不自动改判异常或具体故障"
    return None


def extract_evidence(case, values, *, unit, decomposition=None, modulation=False, dual_envelope=False, robust_periodicity=False, autogram=False,
                     speed_reference=None, on_envelope_spectra=None, config_path=None):
    if decomposition not in (None, "emd", "wavelet"):
        raise ValueError("decomposition must be emd, wavelet or None")
    if type(modulation) is not bool:
        raise ValueError("modulation must be a boolean")
    if type(dual_envelope) is not bool:
        raise ValueError("dual_envelope must be a boolean")
    if type(robust_periodicity) is not bool:
        raise ValueError("robust_periodicity must be a boolean")
    if type(autogram) is not bool:
        raise ValueError("autogram must be a boolean")
    if set(case) != {"signal", "equipment", "operating_condition"}:
        raise ValueError("Extractor accepts only public input, without labels or case identifiers")
    if unit not in ("g", "a.u."):
        raise ValueError("Use confirmed acceleration unit g or explicitly uncalibrated a.u.")
    if case["signal"].get("unit", unit) != unit:
        raise ValueError("Declared signal unit differs from extraction unit")
    base_unit = "g" if unit == "g" else "au"
    x = np.asarray(values, dtype=float)
    fs = float(case["signal"]["sampling_rate_hz"])
    if x.ndim != 1 or len(x) != case["signal"]["segment_samples"] or not np.isfinite(x).all():
        raise ValueError("Signal must be finite, one-dimensional and match segment_samples")
    if not math.isfinite(fs) or fs <= 0:
        raise ValueError("Invalid sampling rate")
    cfg = load_config(config_path)
    matching = cfg["matching"]
    evidence = {"tool_call_chain": [], "extracted_features": [], "frequency_matches": []}

    def call(tool, parameters):
        order = len(evidence["tool_call_chain"])+1
        evidence["tool_call_chain"].append({"order": order, "tool": tool, "parameters": deepcopy(parameters)})
        return order

    def feature(step, name, value):
        evidence["extracted_features"].append({"name": f"step_{step}.{name}", "value": value})

    rpm = case["operating_condition"].get("rpm")
    quality = diagnosability.assess_input_quality(x, fs, rpm, cfg["signal"]["quality"])
    step = call("diagnosability.assess_input_quality", {"sampling_rate_hz": fs, "rpm": rpm, **cfg["signal"]["quality"]})
    if unit == "a.u.":
        evidence["tool_call_chain"][-1]["parameters"]["signal_unit"] = unit
        feature(step, "quality.amplitude_calibration", {
            "unit": unit, "calibrated": False, "conversion_applied": False,
            "limitation": "原始幅值未标定，不是加速度g或已确认电压V；不据此评价绝对振动幅值、跨传感器强弱或严重程度"})
    for key in ("valid_for_analysis", "clipping_ratio", "trend_ratio", "dropout_run_ratio", "invalidating_flags"):
        feature(step, "quality."+key, quality[key])
    feature(step, "raw.duration_seconds", len(x)/fs)
    feature(step, "raw.frequency_grid_hz", fs/max(1, len(x)))
    if not quality["valid_for_analysis"]:
        return evidence
    nominal_rpm = float(rpm)
    speed = speed_reference if speed_reference is not None else estimate_shaft_speed(x, fs, nominal_rpm)
    speed_step = call("speed_estimation.estimate_shaft_speed", {
        "sampling_rate_hz": fs, "nominal_rpm": nominal_rpm, "samples": len(x),
        "input_scope": "current complete input signal only; no source-file lookup or labels",
        **SPEED_SETTINGS})
    feature(speed_step, "shaft.speed_reference", {
        **speed, "duration_seconds": len(x)/fs,
        "estimated_rpm": speed["candidates"][0]["rpm"] if speed["candidates"] else None,
        "reference_scope": "current input signal",
        "used_for_fixed_frequency_analysis": case["operating_condition"].get("operating_state") != "变转速",
    })
    if case["operating_condition"].get("operating_state") == "变转速":
        feature(step, "quality.limitation", "变转速，当前固定频率提取未执行阶次跟踪")
        return evidence
    rpm = speed["rpm"]
    shaft_reference = speed["method"]+" from current input; not measured phase or tachometer rpm"

    time = indicators.time_domain(x, fs)
    step = call("assessment.domain_indicators.time_domain", {"sampling_rate_hz": fs, "rms_segments": 4,
        "kurtosis": "Pearson, population moments", "stationarity_cv_boundaries": [0.15, 0.5],
        "trend_rule": "abs(fitted_slope)*N > 0.1*std; sample-index linear fit"})
    names = {"mean": "mean_g", "std": "std_g", "rms": "rms_g", "peak": "peak_g", "peak_to_peak": "peak_to_peak_g",
             "crest_factor": "crest_factor_ratio", "impulse_factor": "impulse_factor_ratio", "clearance_factor": "clearance_factor_ratio",
             "kurtosis": "kurtosis_pearson", "skewness": "skewness", "modulation_index": "modulation_index_ratio"}
    for key, name in names.items():
        feature(step, "raw."+(name[:-2]+"_"+base_unit if name.endswith("_g") else name), time[key])
    for key, value in time["stationarity"].items():
        feature(step, "raw.stationarity."+("segment_rms_cv_ratio" if key == "segment_rms_cv" else key), value)

    frequency = indicators.frequency_domain(x, fs)
    step = call("assessment.domain_indicators.frequency_domain", {"sampling_rate_hz": fs, "n_peaks": 8,
        "window": "numpy.hanning (symmetric)", "zero_padding": False, "amplitude_unit": base_unit,
        "peak_selection": "largest FFT bins, not independent local-peak detection"})
    feature(step, "fft.dominant_bins", frequency["dominant_peaks"])
    for key in ("spectral_centroid_hz", "spectral_spread_hz", "spectral_entropy", "local_peak_snr_db"):
        feature(step, "fft."+key, frequency[key])

    psd = welch_psd_features(x, fs)
    step = call("features.welch_psd_features", {"sampling_rate_hz": fs, "window": "hann", "nperseg": psd["nperseg"],
        "noverlap": psd["nperseg"]//2, "nfft": psd["nperseg"], "detrend": "constant", "scaling": "density",
        "average": "mean", "reliable_max_hz": psd["reliable_range_hz"][1],
        "bands_hz": [row["band_hz"] for row in psd["band_power_ratios"]]})
    feature(step, "welch.frequency_grid_hz", psd["frequency_resolution_hz"])
    for row in psd["band_power_ratios"]:
        band = "band_"+"_".join(f"{v:g}" for v in row["band_hz"])
        feature(step, f"welch.{band}.power_ratio", row["power_ratio"])
        feature(step, f"welch.{band}.observable", row["observable"])

    try:
        ratios = geometry.fault_frequency_ratios(case["equipment"].get("bearing_geometry"))
        frequencies = geometry.fault_frequencies_hz(rpm, ratios)
    except (ValueError, TypeError) as exc:
        step = call("geometry.fault_frequency_ratios", {"bearing_geometry": case["equipment"].get("bearing_geometry"), "rpm": rpm})
        feature(step, "matching.unavailable_reason", str(exc))
        return evidence
    step = call("geometry.fault_frequency_ratios + geometry.fault_frequencies_hz", {
        "rpm": rpm, "ratios": ratios, "shaft_reference": shaft_reference, "input_step": speed_step})
    for family, value in frequencies.items():
        feature(step, f"theory.{family}.fundamental_hz", value)

    harmonic_count = int(matching["max_harmonics"])
    # Keep the existing band grid so speed correction does not also change band selection.
    min_band = max(50.0, max(ratios[family]*nominal_rpm/60 for family in FAMILIES)*harmonic_count)
    tf = indicators.time_frequency(x, fs, min_band_hz=min_band, n_bands=0 if autogram else 5)
    nperseg = min(1024, max(64, len(x)//16))
    step = call("assessment.domain_indicators.time_frequency", {"sampling_rate_hz": fs, "min_band_hz": min_band, "band_grid_reference_rpm": nominal_rpm, "n_bands": 0 if autogram else 5,
        "stft": {"window": "hann", "nperseg": nperseg, "noverlap": nperseg//2, "nfft": nperseg,
                 "detrend": False, "boundary": "zeros", "padded": True, "scaling": "spectrum"},
        "high_energy_frame_rule": "frame_coefficient_rms > mean + 2*std; counts frames, not impulses",
        "status": tf["stft_status"]})
    feature(step, "stft.available", tf["stft_status"] == "ok")
    if tf["stft_status"] == "ok":
        feature(step, "stft.nonstationarity_index_ratio", tf["nonstationarity_index"])
        feature(step, "stft.high_energy_frame_count", tf["transient_count"])

    processing_input = {}
    if decomposition is not None:
        from .toolbox.contracts import ToolContext
        from .toolbox.registry import build_default_registry

        parameters = {"max_samples": len(x), **({"max_imf": 6} if decomposition == "emd" else {"wavelet": "db4", "level": 4})}
        context = ToolContext(sampling_rate_hz=fs, rpm=rpm, geometry=case["equipment"].get("bearing_geometry"),
                              fault_frequencies_hz=frequencies)
        result = build_default_registry().execute(decomposition+"_decompose", x, context, parameters)
        if result.status != "ok":
            raise RuntimeError(f"{decomposition} decomposition failed: {result.error}")
        processed = np.asarray(result.arrays["signal"], dtype=float)
        if processed.shape != x.shape or not np.isfinite(processed).all() or result.data["processed_samples"] != len(x):
            raise ValueError("Decomposition must preserve the complete finite signal and its sampling rate")
        step = call("toolbox."+result.tool, {"sampling_rate_hz": fs, "rpm": rpm,
            "fault_frequencies_hz": frequencies, **result.params, "algorithm_parameters": result.data["algorithm_parameters"],
            "component_scoring": {"harmonics": 3, "tolerance_hz": "max(fs/4096, target*0.03)",
                "envelope_mode": "squared", "window": "hann", "families": list(frequencies)},
            "selection_rule": "score >= mean(scores)+0.5*std(scores); fallback argmax; sum selected components",
            "input": "complete raw signal; no label or case identifier"})
        feature(step, "decomposition.result", {key: value for key, value in result.data.items() if key != "algorithm_parameters"})
        processing_input = {"input_step": step}
        x = processed

    if autogram:
        from . import autogram_selection

        settings = autogram_settings(frequencies, fs/len(x), matching)
        step = call("autogram.select_resonance_band", {**processing_input, "sampling_rate_hz": fs, **settings,
            "selection_input": "current complete signal; no label or case identifier",
            "frequency_bound": "twice the highest theoretical fundamental across all three bearing families",
            "coverage_budget": "minimum fit order for all families, existing frequency tolerances and filter-edge reserve; not a required peak-detection pattern",
            "downstream_filter": "existing 4th-order Butterworth SOS zero-phase; same band boundaries, different filter shape",
            "failure_policy": "preserve selection failure; no kurtosis or broadband fallback"})
        selection = autogram_selection.select_resonance_band(x, fs, settings)
        feature(step, "selection.autogram", selection)
        if not selection["quality_gate_passed"] or selection.get("selected_band_hz") is None:
            return evidence
        coverage = autogram_coverage(selection["selected_band_hz"], fs, fs/len(x), frequencies, matching)
        feature(step, "selection.coverage", coverage)
        selected = [{"band_hz": selection["selected_band_hz"]}]
        # A single primary band must not be replaced by periodicity's extra-band search.
        scores = []
    else:
        scores = signal_tools.kurtosis_band_scores(x, fs, min_band_hz=min_band, n_edges=10)
        selected = scores[:int(cfg["signal"]["band_count"])]
        step = call("signal_tools.kurtosis_band_scores", {**processing_input, "sampling_rate_hz": fs, "min_band_hz": min_band, "band_grid_reference_rpm": nominal_rpm, "n_edges": 10,
            "n_bands": cfg["signal"]["band_count"], "selection": "top scores including zero/negative scores; no positive-score health gate",
            "score": "excess kurtosis of centered ordinary Hilbert amplitude envelope; not MATLAB kurtogram",
            "filter": "4th-order Butterworth SOS, sosfiltfilt defaults", "candidate_bands_hz": [row["band_hz"] for row in scores]})
        feature(step, "selection.band_count", len(selected))
        for row in scores:
            band = "band_"+"_".join(f"{v:g}" for v in row["band_hz"])
            feature(step, f"selection.{band}.envelope_kurtosis_excess", row["kurtosis"])

    comparisons = []
    modulation_inputs = []
    periodicity_inputs = []
    modes = ("squared", "ordinary") if dual_envelope else (cfg["signal"]["envelope"]["mode"],)
    for row in selected:
        low, high = row["band_hz"]
        band = "band_"+"_".join(f"{v:g}" for v in (low, high))
        kind, cutoff = ("highpass", low/(fs/2)) if high == fs/2 else ("bandpass", [low/(fs/2), high/(fs/2)])
        sos = scipy_signal.butter(4, cutoff, btype=kind, output="sos")
        padlen = 3*(2*len(sos)+1-min(int((sos[:, 2] == 0).sum()), int((sos[:, 5] == 0).sum())))
        filtered = signal_tools.bandpass_filter(x, fs, low, high, order=4)
        filter_step = call("signal_tools.bandpass_filter", {**processing_input, "sampling_rate_hz": fs, "low_hz": low, "high_hz": high,
            "order": 4, "btype": kind, "output": "sos", "sosfiltfilt_padtype": "odd", "sosfiltfilt_padlen": padlen})
        for mode in modes:
            envelope = signal_tools.envelope_spectrum(filtered, fs, mode=mode,
                window=cfg["signal"]["envelope"]["window"], **({"return_envelope": True} if robust_periodicity else {}))
            if autogram:
                envelope["analysis_end_hz"] = coverage["analysis_end_hz"]
            amplitude_unit = base_unit+"2" if envelope["mode"] == "squared" else base_unit
            env_step = call("signal_tools.envelope_spectrum", {"input_step": filter_step, "input_band_hz": [low, high], "band_hz": None,
                "sampling_rate_hz": fs, "mode": envelope["mode"], "window": envelope["window"], "remove_mean": True,
                "trim_samples_each_end": 0, "zero_padding": False, "nfft": len(x), "amplitude_unit": amplitude_unit,
                "spectrum_baseline": "reflect-padded median subtraction, negatives retained",
                "baseline_bins": max(31, min(2*(len(x)//2048)+1, 101) | 1)})
            feature(env_step, f"{band}.rms_{base_unit}", envelope["filtered_rms"])
            feature(env_step, f"{band}.envelope.frequency_grid_hz", envelope["frequency_resolution_hz"])
            feature(env_step, f"{band}.envelope.peak_to_p95_db", envelope["envelope_snr_db"])
            if modulation:
                modulation_inputs.append((env_step, [low, high], envelope))
            if robust_periodicity:
                periodicity_inputs.append((env_step, [low, high], envelope, True))
            df = envelope["frequency_resolution_hz"]
            match_steps, match_results = {}, {}
            for family in FAMILIES:
                fundamental = frequencies[family]
                tolerance = lambda harmonic, family=family: _tolerance(family, frequencies[family], harmonic, df, matching)
                result = matcher.match_harmonics(envelope["frequencies_hz"], envelope["amplitudes"], fundamental,
                    max_harmonics=harmonic_count, tolerance_hz_fn=tolerance,
                    min_relative_amplitude=matching["min_relative_amplitude"], min_robust_z=matching["min_robust_z"],
                    background_radius_bins=matching["background_radius_bins"])
                match_step = call("matcher.match_harmonics", {"input_step": env_step, "band_hz": [low, high], "family": family,
                    **({"envelope_mode": mode, "amplitude_unit": amplitude_unit} if dual_envelope else {}),
                    "fundamental_hz": fundamental, "frequency_grid_hz": df, **matching,
                    "tolerance_formula": "max(resolution_multiplier*df, minimum_hz, target*sqrt(speed_u^2+geometry_u^2+slip_u^2))",
                    "observability": ("conservative centered-carrier span with FFT filter-edge reserve; not physical absence"
                                      if autogram else "target+tolerance < carrier_bandwidth/2 and target+tolerance < Nyquist"),
                    **({"analysis_end_hz": envelope["analysis_end_hz"]} if autogram else {})})
                match_steps[family] = match_step
                match_results[family] = result["harmonic_peaks"]
                peaks = {peak["harmonic"]: peak for peak in result["harmonic_peaks"]}
                for harmonic in range(1, harmonic_count+1):
                    target, tol = fundamental*harmonic, tolerance(harmonic)
                    peak = peaks.get(harmonic)
                    observable = target+tol < envelope.get("analysis_end_hz", min((high-low)/2, fs/2))
                    actual_peak = bool(observable and peak and peak["is_local_peak"] and peak["amplitude"] > 0)
                    if not observable:
                        assessment = "不可核验：解调带宽或频率范围不足"
                    elif not actual_peak:
                        assessment = "未识别到容差内高于所减谱底的局部峰"
                    elif peak["qualified"]:
                        assessment = "满足当前峰值筛选条件；故障归属由专家判读"
                    else:
                        assessment = "存在候选峰，但未满足当前幅值或局部背景筛选条件"
                    if actual_peak and abs(peak["observed_frequency_hz"]-round(peak["observed_frequency_hz"]/(rpm/60))*(rpm/60)) <= 2*df:
                        assessment += "；邻近转频整数倍"
                    evidence["frequency_matches"].append({"family": family, "harmonic": harmonic, "source_step": match_step,
                        "theoretical_hz": target, "observed_hz": peak["observed_frequency_hz"] if actual_peak else None,
                        "tolerance_hz": tol, "assessment": assessment})
                    prefix = f"{band}.envelope.{family}.h{harmonic}"
                    feature(match_step, prefix+".observable", observable)
                    if observable and peak:
                        feature(match_step, prefix+".peak_amplitude_"+amplitude_unit, peak["amplitude"] if actual_peak else None)
                        feature(match_step, prefix+".relative_amplitude_ratio", peak["relative_amplitude"] if actual_peak else None)
                        feature(match_step, prefix+".local_robust_z_ratio", peak["local_robust_z"] if actual_peak else None)
                    if actual_peak:
                        observed = peak["observed_frequency_hz"]
                        shaft_multiple = max(1, round(observed/(rpm/60)))
                        feature(match_step, prefix+".distance_to_fault_hz", abs(observed-target))
                        feature(match_step, prefix+".nearest_shaft_multiple", shaft_multiple)
                        feature(match_step, prefix+".nearest_shaft_harmonic_hz", shaft_multiple*(rpm/60))
                        feature(match_step, prefix+".distance_to_shaft_hz", abs(observed-shaft_multiple*(rpm/60)))
            comparisons.append((band, [low, high], envelope, match_steps, match_results))

    # Compare after collecting every original match, preserving their source-step IDs.
    for band, bounds, envelope, match_steps, match_results in comparisons:
        df = envelope["frequency_resolution_hz"]
        end_hz = max(0.0, envelope.get("analysis_end_hz", min((bounds[1]-bounds[0])/2, fs/2))-LATTICE_SETTINGS["peak_half_bins"]*df)
        for family in FAMILIES:
            result = matcher.lattice_evidence(envelope["frequencies_hz"], envelope["amplitudes"], frequencies[family],
                max_harmonics=harmonic_count, shaft_hz=rpm/60, analysis_end_hz=end_hz,
                background_radius_bins=matching["background_radius_bins"], **LATTICE_SETTINGS)
            step = call("matcher.lattice_evidence", {"input_match_step": match_steps[family], "band_hz": bounds,
                **({"envelope_mode": envelope["mode"]} if dual_envelope else {}),
                "family": family, "fundamental_hz": frequencies[family], "shaft_hz": rpm/60,
                "shaft_reference": shaft_reference,
                "max_harmonics": harmonic_count, "analysis_end_hz": end_hz,
                "background_radius_bins": matching["background_radius_bins"], **LATTICE_SETTINGS,
                "attribution_check": {"tolerance_source": "same harmonic_tolerance_hz as strict matching",
                    "frequency_rule": "abs(fit-theory) <= min(tolerance(h)/h) over retained visible orders",
                    "same_peak_tolerance_hz": df/2, "strict_min_relative_amplitude": matching["min_relative_amplitude"],
                    "strict_min_robust_z": matching["min_robust_z"],
                    "minimum_corresponding_harmonics": LATTICE_SETTINGS["min_orders_for_fit"]},
                "purpose": "compare pure fault harmonic lattice against shaft harmonics; does not filter samples"})
            prefix = f"{band}.comparison.{family}"
            fault = result["models"]["lattice"]
            shaft = result["models"].get("shaft", {})
            feature(step, prefix+".best_model", result["best_model"])
            feature(step, prefix+".fitted_is_interference", result["fitted_is_interference"])
            feature(step, prefix+".accept_threshold_bins", result["accept_threshold_bins"])
            for model_name, model in (("fault", fault), ("shaft", shaft)):
                for key in ("residual_rms_bins", "visible_line_count", "line_count"):
                    value = model.get(key)
                    feature(step, prefix+f".{model_name}."+key, value if value is not None and math.isfinite(value) else None)
            for key, value in (("fault.fitted_frequency_hz", fault.get("fitted_frequency_hz")),
                               ("shaft.fitted_frequency_hz", shaft.get("fitted_shaft_hz"))):
                feature(step, prefix+"."+key, value if value is not None and math.isfinite(value) else None)
            observable = {h for h in range(1, harmonic_count+1)
                if frequencies[family]*h+_tolerance(family, frequencies[family], h, df, matching)
                < envelope.get("analysis_end_hz", min((bounds[1]-bounds[0])/2, fs/2))}
            verification = verify_lattice(result, family, frequencies[family], df, matching,
                                          match_results[family], observable)
            feature(step, prefix+".verification", verification)
            note = "；"+verification["assessment"]
            for match in evidence["frequency_matches"]:
                if match["source_step"] == match_steps[family]:
                    match["assessment"] += note+f"（步骤{step}；整体谱列比较，非逐峰来源证明）"
    if modulation:
        from . import modulation_evidence as mod

        step = call("modulation.symmetric_sidebands", {"input_steps": [item[0] for item in modulation_inputs],
            "input_bands_hz": [item[1] for item in modulation_inputs], "sampling_rate_hz": fs,
            **({"input_envelope_modes": [item[2]["mode"] for item in modulation_inputs],
                "amplitude_units": {"squared": base_unit+"2", "ordinary": base_unit},
                "cross_band_rule": "confirmation must use the same envelope mode; modes are not independent measurements"}
               if dual_envelope else {"amplitude_unit": base_unit+"2" if modes[0] == "squared" else base_unit}),
            "frequency_grid_hz": fs/len(x), "shaft_hz": rpm/60, "shaft_reference": shaft_reference,
            "fundamentals_hz": {family: frequencies[family] for family in FAMILIES}, **mod.SETTINGS,
            "min_relative_amplitude": matching["min_relative_amplitude"], "min_robust_z": matching["min_robust_z"],
            "background_radius_bins": matching["background_radius_bins"],
            "shaft_speed_relative_uncertainty": matching["speed_relative_uncertainty"],
            "shaft_separation_rule": "distance > max(shaft_exclusion_bins*df, observed_hz*shaft_speed_relative_uncertainty)",
            "carrier_tolerance": "same harmonic_tolerance_hz as original strict matching",
            "observability": "all sideband windows below carrier bandwidth/2 and Nyquist",
            "purpose": "confirm modulation structure; not a fault localization verdict"})
        groups = []
        for _, bounds, envelope in modulation_inputs:
            end_hz = envelope.get("analysis_end_hz", min((bounds[1]-bounds[0])/2, fs/2))
            for family in FAMILIES:
                found = mod.symmetric_groups(envelope["frequencies_hz"], envelope["amplitudes"], family=family,
                    fundamental_hz=frequencies[family], shaft_hz=rpm/60,
                    tolerance_hz_fn=lambda h, family=family: _tolerance(family, frequencies[family], h, fs/len(x), matching),
                    analysis_end_hz=end_hz, min_relative_amplitude=matching["min_relative_amplitude"],
                    min_robust_z=matching["min_robust_z"], background_radius_bins=matching["background_radius_bins"],
                    shaft_speed_relative_uncertainty=matching["speed_relative_uncertainty"])
                groups.extend({**g, "band_hz": bounds,
                               **({"envelope_mode": envelope["mode"], "amplitude_unit": base_unit+"2" if envelope["mode"] == "squared" else base_unit}
                                  if dual_envelope else {})} for g in found)
        feature(step, "modulation.groups", groups)
        summaries = {family: mod.cross_band_summary(groups, family, fs/len(x)) for family in FAMILIES}
        for summary in summaries.values():
            summary["shaft_reference"] = shaft_reference
        if autogram:
            for family, summary in summaries.items():
                summary.update(supported=None, cross_band_available=False,
                    within_band_groups=[g for g in groups if g["family"] == family],
                    assessment="只有一个主频带，跨频带支持不可核验；保留单频带对称边带候选，不把它升级为原跨频带支持，也不据此判断无调制或正常")
        feature(step, "modulation.summary", summaries)
    if robust_periodicity:
        from . import periodicity_evidence as periodicity

        selected_bounds = {tuple(row["band_hz"]) for row in selected}
        for row in scores:
            low, high = row["band_hz"]
            if (low, high) in selected_bounds:
                continue
            filtered = signal_tools.bandpass_filter(x, fs, low, high, order=4)
            filter_step = call("periodicity.additional_band_filter", {**processing_input, "sampling_rate_hz": fs,
                "band_hz": [low, high], "order": 4, "filter": "signal_tools.bandpass_filter; Butterworth SOS zero-phase defaults"})
            for mode in modes:
                envelope = signal_tools.envelope_spectrum(filtered, fs, mode=mode,
                    window=cfg["signal"]["envelope"]["window"], return_envelope=True)
                env_step = call("periodicity.additional_band_envelope", {"input_step": filter_step,
                    "band_hz": [low, high], "sampling_rate_hz": fs, "mode": mode,
                    "window": envelope["window"], "nfft": len(x), "zero_padding": False,
                    "spectrum_baseline": "same median subtraction as signal_tools.envelope_spectrum",
                    "amplitude_unit": base_unit+"2" if mode == "squared" else base_unit})
                periodicity_inputs.append((env_step, [low, high], envelope, False))
        period_step = call("periodicity.discover_and_assess", {"input_steps": [v[0] for v in periodicity_inputs],
            "sampling_rate_hz": fs, "rpm": rpm, "shaft_reference": shaft_reference, "bearing_frequencies_hz": frequencies,
            "matching": matching, "settings": periodicity.SETTINGS,
            "discovery_input": "current envelope spectrum only; no theory, rpm or labels",
            "assessment": "complete-envelope halves; Hann FFT, median baseline subtraction; autocorrelation; fixed models using adopted speed on same peaks",
            "purpose": "periodicity observations, not a health or fault verdict"})
        records = []
        for env_step, bounds, envelope, main_band in periodicity_inputs:
            end_hz = envelope.get("analysis_end_hz", min((bounds[1]-bounds[0])/2, fs/2))
            discovery = periodicity.discover_periods(envelope["frequencies_hz"], envelope["amplitudes"], analysis_end_hz=end_hz)
            measurements = {}
            for candidate in discovery["candidates"]:
                candidate["assessment"] = periodicity.assess_period_candidate(candidate,
                    envelope_values=envelope["envelope_values"], sampling_rate_hz=fs,
                    analysis_end_hz=end_hz, rpm=rpm, bearing_frequencies=frequencies, matching=matching,
                    _measurements=measurements)
                candidate["assessment"]["shaft_reference"] = shaft_reference
                unit_key = base_unit+"2" if envelope["mode"] == "squared" else base_unit
                candidate["summed_peak_amplitude_"+unit_key] = candidate.pop("strength")
                for peak in candidate["peaks"]:
                    peak["peak_amplitude_"+unit_key] = peak.pop("amplitude")
            records.append({"source_step": env_step, "band_hz": bounds, "envelope_mode": envelope["mode"],
                "amplitude_unit": base_unit+"2" if envelope["mode"] == "squared" else base_unit,
                "in_original_top_bands": None if autogram else main_band, "frequency_grid_hz": fs/len(x), "discovery": discovery})
        feature(period_step, "periodicity.candidates", records)
        cross_band = periodicity.cross_band_summary(records, fs/len(x))
        if autogram:
            cross_band.update(available=False, assessment="只有一个主频带，未执行跨频带重复核验；不表示没有周期或正常")
        feature(period_step, "periodicity.cross_band", cross_band)
    if on_envelope_spectra is not None:
        on_envelope_spectra([
            {"band_id": band, "band_hz": bounds, "frequencies_hz": spectrum["frequencies_hz"],
             "amplitudes": spectrum["amplitudes"],
             "families": {family: {"harmonic_peaks": [
                 {"qualified": row["assessment"].startswith("满足当前峰值筛选条件"), "observed_frequency_hz": row["observed_hz"]}
                 for row in evidence["frequency_matches"]
                 if row["source_step"] == step and row["observed_hz"] is not None]}
                 for family, step in steps.items()}}
            for band, bounds, spectrum, steps, _ in comparisons
        ], frequencies)
    return evidence


def compact_evidence(evidence):
    """Summarize completed calculations without using a label or diagnosis."""
    calls = evidence["tool_call_chain"]
    calls_by_order = {call["order"]: call for call in calls}
    groups = {}
    for call in calls:
        groups.setdefault(call["tool"], []).append(call)
    steps = {call["order"]: order for order, group in enumerate(groups.values(), 1) for call in group}
    compact = {"tool_call_chain": [], "extracted_features": [], "frequency_matches": []}
    for order, (tool, group) in enumerate(groups.items(), 1):
        parameters = []
        for call in group:
            params = deepcopy(call["parameters"])
            for key in ("input_step", "input_match_step"):
                if key in params:
                    params[key] = steps[params[key]]
            if "input_steps" in params:
                params["input_steps"] = [steps[value] for value in params["input_steps"]]
            parameters.append(params)
        common = {key: value for key, value in parameters[0].items()
                  if all(key in params and params[key] == value for params in parameters)}
        variants = [{key: value for key, value in params.items() if key not in common} for params in parameters]
        if any(variants):
            common["calls"] = variants
        compact["tool_call_chain"].append({"order": order, "tool": tool, "parameters": common})

    by_step = {}
    for row in evidence["extracted_features"]:
        step, name = row["name"].split(".", 1)
        by_step.setdefault(int(step[5:]), {})[name] = row["value"]

    def feature(step, name, value):
        compact["extracted_features"].append({"name": f"step_{steps[step]}.{name}", "value": deepcopy(value)})

    for call in calls:
        old = call["order"]
        values = by_step.get(old, {})
        if call["tool"] == "diagnosability.assess_input_quality":
            feature(old, "quality", {name[8:]: value for name, value in values.items() if name.startswith("quality.")})
            feature(old, "raw.duration_seconds", values["raw.duration_seconds"])
            feature(old, "raw.frequency_grid_hz", values["raw.frequency_grid_hz"])
        elif call["tool"] == "assessment.domain_indicators.time_domain":
            for name, value in values.items():
                feature(old, name, value)
        elif call["tool"] == "speed_estimation.estimate_shaft_speed":
            feature(old, "shaft.speed_reference", values["shaft.speed_reference"])
        elif call["tool"] == "assessment.domain_indicators.frequency_domain":
            for name, value in values.items():
                feature(old, name, value)
        elif call["tool"] == "features.welch_psd_features":
            rows = []
            for name, value in values.items():
                band = re.fullmatch(r"welch.band_([^_]+)_([^_]+).power_ratio", name)
                if band:
                    rows.append({"band_hz": [float(edge) for edge in band.groups()], "power_ratio": value,
                                 "observable": values[name.replace(".power_ratio", ".observable")]})
            feature(old, "welch.band_power_ratios", rows)
            feature(old, "welch.frequency_grid_hz", values["welch.frequency_grid_hz"])
        elif call["tool"] == "geometry.fault_frequency_ratios + geometry.fault_frequencies_hz":
            feature(old, "theory.fundamentals_hz", {family: values[f"theory.{family}.fundamental_hz"]
                                                    for family in FAMILIES})
        elif call["tool"] == "assessment.domain_indicators.time_frequency":
            for name, value in values.items():
                feature(old, name, value)
        elif call["tool"] == "periodicity.discover_and_assess":
            records = deepcopy(values["periodicity.candidates"])
            for record in records:
                record["source_step"] = steps[record["source_step"]]
            feature(old, "periodicity.candidates", records)
            feature(old, "periodicity.cross_band", values["periodicity.cross_band"])
        elif call["tool"] == "signal_tools.kurtosis_band_scores":
            rows = []
            for band in call["parameters"]["candidate_bands_hz"]:
                name = "selection.band_"+"_".join(f"{edge:g}" for edge in band)+".envelope_kurtosis_excess"
                rows.append({"band_hz": band, "envelope_kurtosis_excess": values[name]})
            feature(old, "selection.band_scores", rows)
        elif call["tool"] == "autogram.select_resonance_band":
            feature(old, "selection.autogram", values["selection.autogram"])
            if "selection.coverage" in values:
                feature(old, "selection.coverage", values["selection.coverage"])
        elif call["tool"] in ("toolbox.emd_decompose", "toolbox.wavelet_decompose"):
            feature(old, "decomposition.result", values["decomposition.result"])
        elif call["tool"] == "modulation.symmetric_sidebands":
            feature(old, "modulation.summary", values["modulation.summary"])

    envelopes = [call for call in calls if call["tool"] == "signal_tools.envelope_spectrum"]
    multiple_modes = len({call["parameters"]["mode"] for call in envelopes}) > 1
    if envelopes:
        rows = []
        for call in envelopes:
            values = by_step[call["order"]]
            rms_name = "rms_au" if call["parameters"]["amplitude_unit"].startswith("au") else "rms_g"
            rows.append({"band_hz": call["parameters"]["input_band_hz"],
                **({"envelope_mode": call["parameters"]["mode"]} if multiple_modes else {}),
                "frequency_grid_hz": next(value for name, value in values.items() if name.endswith(".frequency_grid_hz")),
                rms_name: next(value for name, value in values.items() if name.endswith("."+rms_name)),
                "peak_to_p95_db": next(value for name, value in values.items() if name.endswith(".peak_to_p95_db"))})
        feature(envelopes[0]["order"], "envelope.bands", rows)

    matches_by_step = {}
    for row in evidence["frequency_matches"]:
        matches_by_step.setdefault(row["source_step"], []).append(row)
    comparisons = {}
    family_rows = {family: [] for family in FAMILIES}
    comparison_step = None
    for call in calls:
        if call["tool"] != "matcher.lattice_evidence":
            continue
        comparison_step = call["order"]
        params = call["parameters"]
        match_step, family = params["input_match_step"], params["family"]
        values = {name.split(f".comparison.{family}.", 1)[1]: value
                  for name, value in by_step[call["order"]].items()}
        comparisons[match_step] = values
        coverage = matches_by_step[match_step]
        harmonics = []
        for row in coverage:
            prefix = f".envelope.{family}.h{row['harmonic']}."
            metrics = {name.split(prefix, 1)[1]: value for name, value in by_step[match_step].items() if prefix in name}
            harmonics.append({key: deepcopy(row[key]) for key in
                              ("harmonic", "theoretical_hz", "observed_hz", "tolerance_hz")})
            harmonics[-1].update({"qualified": row["assessment"].startswith("满足"),
                "participates_in_support": bool(values["verification"]["fault_specific_support"]
                    and row["harmonic"] in values["verification"]["corresponding_harmonics"]),
                **metrics})
        verification = deepcopy(values["verification"])
        for name, value in harmonic_support_pattern(verification["corresponding_harmonics"]).items():
            verification.setdefault(name, value)
        family_rows[family].append({"family": family, "source_step": steps[match_step],
            "band_hz": params["band_hz"],
            "envelope_mode": calls_by_order[calls_by_order[match_step]["parameters"]["input_step"]]["parameters"]["mode"],
            "harmonics": harmonics,
            "accept_threshold_bins": values["accept_threshold_bins"],
            "fault_line_count": values["fault.line_count"], "shaft_line_count": values["shaft.line_count"],
            "observable_harmonics": [row["harmonic"] for row in coverage if not row["assessment"].startswith("不可核验")],
            "qualified_harmonics": [row["harmonic"] for row in coverage if row["assessment"].startswith("满足")],
            "best_model": values["best_model"], "fitted_is_interference": values["fitted_is_interference"],
            "fault_residual_bins": values["fault.residual_rms_bins"],
            "shaft_residual_bins": values["shaft.residual_rms_bins"],
            "fault_visible_lines": values["fault.visible_line_count"],
            "shaft_visible_lines": values["shaft.visible_line_count"],
            "fault_fitted_hz": values["fault.fitted_frequency_hz"],
            "shaft_fitted_hz": values["shaft.fitted_frequency_hz"],
            "verification": verification})
    if comparison_step is not None:
        for family, rows in family_rows.items():
            feature(comparison_step, f"comparison.{family}", rows)

    def peak_metrics(row):
        prefix = f".envelope.{row['family']}.h{row['harmonic']}."
        return {name.split(prefix, 1)[1]: value for name, value in by_step.get(row["source_step"], {}).items()
                if prefix in name and name.split(prefix, 1)[1] in
                {"peak_amplitude_g2", "peak_amplitude_g", "peak_amplitude_au2", "peak_amplitude_au", "relative_amplitude_ratio", "local_robust_z_ratio"}}

    for family in FAMILIES:
        rows = [row for row in evidence["frequency_matches"] if row["family"] == family]
        peaks = [row for row in rows if row["observed_hz"] is not None]
        peaks.sort(key=lambda row: (row["assessment"].startswith("满足"),
                   peak_metrics(row).get("local_robust_z_ratio") or 0), reverse=True)
        selected = []
        # Keep a periodic candidate and a competing explanation, including rejected attribution.
        for specific in (True, False):
            candidate = next((row for row in peaks if
                (comparisons.get(row["source_step"], {}).get("best_model") == "lattice" and
                 not comparisons.get(row["source_step"], {}).get("fitted_is_interference", False)) == specific), None)
            if candidate is not None:
                selected.append(candidate)
        for row in peaks:
            if len(selected) >= 2:
                break
            if not any((r["harmonic"], r["observed_hz"]) == (row["harmonic"], row["observed_hz"]) for r in selected):
                selected.append(row)
        if not selected and rows:
            selected = [next((row for row in rows if not row["assessment"].startswith("不可核验")), rows[0])]
        for row in selected:
            match_step = row["source_step"]
            params = next(call["parameters"] for call in calls if call["order"] == match_step)
            band = params["band_hz"]
            comparison = comparisons.get(match_step, {})
            assessment = row["assessment"].split("；", 1)[0]
            verification = comparison["verification"]
            assessment += "；"+verification["assessment"]
            if verification["fault_specific_support"] and row["harmonic"] not in verification["corresponding_harmonics"]:
                assessment += "；本峰未参与上述同组严格匹配支持"
            compact["frequency_matches"].append({**row, "source_step": steps[match_step],
                "assessment": (("普通包络；" if params["envelope_mode"] == "ordinary" else "平方包络；") if multiple_modes else "")
                              +f"频带{band[0]:.2f}–{band[1]:.2f}Hz；"+assessment})
            metrics = peak_metrics(row)
            if metrics:
                feature(match_step, f"peak_{len(compact['frequency_matches'])}", {"band_hz": band,
                        **({"envelope_mode": params["envelope_mode"]} if multiple_modes else {}), **metrics})
    return compact


def round_evidence(value, key=""):
    """Round saved evidence only; calculations and expert inputs keep full precision."""
    if isinstance(value, dict):
        return {name: round_evidence(child, value.get("name", "") if name == "value" else name)
                for name, child in value.items()}
    if isinstance(value, list):
        return [round_evidence(child, key) for child in value]
    if isinstance(value, float):
        if key.endswith(("_hz", "_db", "_rpm")) or key == "rpm":
            rounded = round(value, 2)
        elif key.endswith("_bins"):
            rounded = round(value, 3)
        else:
            rounded = float(f"{value:.4g}")
        return int(rounded) if rounded.is_integer() else rounded
    return value


def classification_support(evidence, fault_type):
    """Check a disputed diagnosis against recorded support patterns, without labels."""
    family = FAULT_TO_FAMILY.get(fault_type)
    if family not in FAMILIES:
        return None
    patterns = []
    for row in evidence.get("extracted_features", []):
        if row.get("name", "").endswith(".comparison."+family) and isinstance(row.get("value"), list):
            for band in row["value"]:
                verification = band.get("verification", {})
                if verification.get("fault_specific_support"):
                    patterns.append(verification.get("support_pattern"))
    known = all(pattern is not None for pattern in patterns)
    anchored = sum(pattern == "anchored_contiguous" for pattern in patterns)
    sparse = sum(pattern == "sparse" for pattern in patterns)
    if not known:
        assessment = None
    elif anchored:
        assessment = "所判故障族存在低阶连续倍频支持，分类疑点保留供人工复核"
    elif sparse:
        assessment = "所判故障族仅有稀疏候选支持，缺少低阶连续倍频，维持待复核"
    else:
        assessment = "所判故障族未形成倍频支持，维持待复核"
    return {"family": family, "supported_bands": len(patterns), "anchored_bands": anchored,
            "sparse_bands": sparse, "pattern_known": known, "assessment": assessment}


def validate_analysis(analysis, evidence):
    errors = []
    if not isinstance(analysis, dict) or set(analysis) != {"fault_type", "reason"}:
        return ["输出必须且仅包含 fault_type、reason"]
    if not isinstance(analysis["fault_type"], str) or analysis["fault_type"] not in FAULT_NAMES or not isinstance(analysis["reason"], str):
        return ["类别或文本类型不正确"]
    reason = analysis["reason"]
    if analysis["fault_type"] == "normal":
        coverage_issue = normal_coverage_issue(evidence)
        if coverage_issue:
            errors.append(coverage_issue)
    if analysis["fault_type"] == "normal" and any(
        pair["both_half_windows_repeat"] and pair["both_separated_from_nominal_shaft"] and pair["compatible_families"]
        for row in evidence["extracted_features"] if row["name"].endswith(".periodicity.cross_band") for pair in row["value"]["pairs"]):
        errors.append("当前片段存在跨频带、半窗重复且与故障频率相容的周期候选，正常依据需复核；不自动改判异常或具体部位")
    if analysis["fault_type"] == "normal" and any(
        any(value["supported"] for value in row["value"].values())
        for row in evidence["extracted_features"] if row["name"].endswith(".modulation.summary")):
        errors.append("已有跨频带对称调制支持，正常结论需复核；不能仅凭故障定位不足判正常")
    if re.search(r"未知|\bunknown\b", reason, re.I):
        errors.append("分析文本不使用未知结论；已确认异常但不能定位时应说明异常表现及定位限制")
    if not 300 <= len("".join(reason.split())) <= 500:
        errors.append("分析文本须为300—500个非空白字符")
    coded_terms = re.findall(r"(?<![A-Za-z0-9_])(?:[A-Za-z][A-Za-z0-9]*_[A-Za-z0-9_]+|lattice|shaft|none|true|false)(?![A-Za-z0-9_])|晶格|通约|最佳模型胜出|最佳解释模型", reason, re.I)
    if coded_terms:
        errors.append("分析正文需用中文解释证据，不能展示内部字段名或模型枚举："+"、".join(dict.fromkeys(coded_terms)))
    positions = [reason.find(heading) for heading in HEADINGS]
    if any(reason.count(heading) != 1 for heading in HEADINGS) or positions != sorted(set(positions)):
        errors.append("分析文本必须按顺序包含四段标题")
    cited = {int(n) for group in re.findall(r"步骤\s*(\d+(?:\s*[、,，]\s*\d+)*)", reason) for n in re.findall(r"\d+", group)}
    known = {row["order"] for row in evidence["tool_call_chain"]}
    if not cited or cited-known:
        errors.append("缺少步骤引用或引用了不存在的步骤")
    conclusion = reason.split(HEADINGS[-1])[-1]
    markers = re.findall(r"(?:诊断|判断|判定|结论)(?:结果)?(?:为|是|：|:)[\s“\"：:]*"
                         r"(内圈故障|外圈故障|滚动体故障|正常|异常|未知|unknown)", conclusion)
    expected = FAULT_NAMES[analysis["fault_type"]]
    if not markers or any(label != expected for label in markers):
        errors.append("综合判断需明确写‘判断为：类别’，且与 fault_type 一致")
    errors.extend(validate_quantities(reason, evidence))
    return errors


def validate_quantities(reason, evidence):
    """Check recorded quantities, not free-text causality or statistical significance."""
    pool = {unit: [] for unit in ("hz", "g", "g2", "au", "au2", "V", "V2", "db", "rpm", "s")}
    fft_unit = next((call.get("parameters", {}).get("amplitude_unit", "g") for call in evidence.get("tool_call_chain", [])
                     if call.get("tool") == "assessment.domain_indicators.frequency_domain"), "g")

    def collect(value, key=""):
        if isinstance(value, dict):
            if set(value) == {"name", "value"}:
                band = re.search(r"\.welch\.band_([^_]+)_([^_]+)\.(?:power_ratio|observable)$", value["name"])
                if band:
                    pool["hz"].extend(float(edge) for edge in band.groups())
                if value["name"].endswith("fft.dominant_bins"):
                    for row in value["value"]:
                        collect(row["amplitude"], "amplitude_"+fft_unit)
                        collect(row["frequency_hz"], "frequency_hz")
                else:
                    collect(value["value"], value["name"])
            else:
                for name, child in value.items():
                    collect(child, name)
        elif isinstance(value, list):
            for child in value:
                collect(child, key)
        elif isinstance(value, (float, int)) and not isinstance(value, bool):
            unit = next((u for suffix, u in (("_hz", "hz"), ("_g2", "g2"), ("_g", "g"), ("_au2", "au2"), ("_au", "au"), ("_db", "db"),
                                            ("_seconds", "s"), ("rpm", "rpm")) if key.endswith(suffix)), None)
            if unit:
                pool[unit].append(float(value))
                if unit == "rpm":
                    pool["hz"].append(float(value)/60)
    collect(evidence)
    pattern = r"(?<![A-Za-z\d.])([-+]?(?:\d+(?:\.\d+)?|\.\d+)(?:[eE][-+]?\d+)?)\s*(kHz|Hz|g²|g2|g\^2|g|a\.u\.²|a\.u\.\^2|a\.u\.|au2|au|V²|V\^2|V2|V|dB|rpm|r/min|秒|s)(?![A-Za-z])"
    errors = []
    for match in re.finditer(pattern, reason):
        token, unit = match.groups()
        value = float(token)
        mantissa, _, exponent = token.lower().partition("e")
        rounding = 10**(int(exponent or "0")-len(mantissa.partition(".")[2]))
        if unit == "kHz":
            value, rounding, unit = value*1000, rounding*1000, "hz"
        unit = {"Hz": "hz", "g²": "g2", "g^2": "g2", "a.u.": "au", "a.u.²": "au2", "a.u.^2": "au2", "V²": "V2", "V^2": "V2", "dB": "db", "r/min": "rpm", "秒": "s"}.get(unit, unit)
        if not any(abs(value-recorded) <= rounding*0.501+abs(recorded)*1e-12 for recorded in pool[unit]):
            errors.append("正文量值未能与记录核对: "+match.group())
    return list(dict.fromkeys(errors))
