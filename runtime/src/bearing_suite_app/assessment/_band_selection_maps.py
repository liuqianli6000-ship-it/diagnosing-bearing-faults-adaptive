"""Filter-bank scores ported from bearing_diagnosis_agent/benchmark/band_selection.py.

Third-party notices are in the portable Skill references directory.
"""
from __future__ import annotations

import numpy as np
import pywt
from scipy import signal


def spectral_negentropy(values):
    """I=mean(r log r), r=|values|²/mean(|values|²)，自然对数，0log0=0。"""
    magnitude = np.abs(values)
    scale = np.max(magnitude)
    if scale == 0:
        return 0.
    power = (magnitude/scale)**2
    ratio = power/np.mean(power)
    positive = ratio > 0
    return float(np.sum(ratio[positive]*np.log(ratio[positive]))/len(ratio))


def autocorrelation_kurtosis(squared_envelope):
    """原始Autogram（非Upper/Lower）：无偏AC，取1%至55%正延迟，减最小值后矩比。"""
    x = np.asarray(squared_envelope, dtype=float)
    scale = np.max(np.abs(x))
    if scale == 0:
        return 0.
    x = x/scale
    n = len(x)
    first, last = int(.01*n)+1, int(.55*n)
    ac = signal.correlate(x, x, mode='full', method='fft')[n-1+first:n-1+last]
    ac = ac/(n-np.arange(first, last))
    ac -= ac.min()
    scale = np.max(ac)
    if scale <= 1e-12:
        return 0.
    ac /= scale
    return float(np.mean(ac**4)/np.mean(ac**2)**2)


def _complex_kurtosis(values):
    x = values-np.mean(values)
    scale = np.max(np.abs(x))
    if scale == 0:
        return 0.
    power = np.abs(x/scale)**2
    return float(np.mean(power**2)/np.mean(power)**2 - (2 if np.iscomplexobj(values) else 3))


def _cell(low, high, score, **extra):
    return {'band_hz': [float(low), float(high)], 'score': float(score), **extra}


def _finish(rows, has_energy):
    cells = [{**c, 'level': row['level']} for row in rows for c in row['cells']]
    ranked = sorted(cells, key=lambda c: -c['score'])
    return {'rows': rows, 'recommended_band_hz': ranked[0]['band_hz'] if has_energy else None,
            'top_candidates': ranked[:5] if has_energy else [], 'has_signal_energy': has_energy}


def _fast_kurtogram(x, fs, depth):
    # Antoni的短解析FIR生成器；二叉节点调制恢复频率顺序，三叉层插在整数层之间。
    n = np.arange(17)
    h = signal.firwin(17, .4, window='hamming')*np.exp(2j*np.pi*n*.125)
    m = np.arange(1, 17)
    g = h[(-m) % 16]*(-1.)**m
    n3 = np.arange(25)
    base = signal.firwin(25, .4*2/3, window='hamming')
    triple = [base*np.exp(2j*np.pi*n3*(1/12+k/6)) for k in range(3)]
    rows = [{'level': 0., 'cells': [_cell(0, fs/2, max(0., _complex_kurtosis(x)))]}]
    nodes = [(x, 0)]
    for parent_depth in range(depth):
        count = len(nodes)
        if parent_depth <= depth-2:
            cells = []
            for index, (node, trim) in enumerate(nodes):
                for k, filt in enumerate(triple):
                    child = signal.lfilter(filt, [1.], node)[2::3]
                    start = int(np.ceil((trim+len(filt)-1)/3))
                    low, high = (index+k/3)*fs/(2*count), (index+(k+1)/3)*fs/(2*count)
                    cells.append(_cell(low, high, max(0., _complex_kurtosis(child[start:]))))
            rows.append({'level': float(parent_depth+np.log2(3)), 'cells': cells})
        children, cells = [], []
        for index, (node, trim) in enumerate(nodes):
            for k, filt in enumerate((h, g)):
                child = signal.lfilter(filt, [1.], node)[1::2]
                if k:
                    child = child*(-1.)**np.arange(1, len(child)+1)
                start = int(np.ceil((trim+len(filt)-1)/2))
                children.append((child, start))
                low, high = (2*index+k)*fs/(4*count), (2*index+k+1)*fs/(4*count)
                cells.append(_cell(low, high, max(0., _complex_kurtosis(child[start:]))))
        rows.append({'level': float(parent_depth+1), 'cells': cells})
        nodes = children
    return sorted(rows, key=lambda r: r['level'])


def _modwpt_rows(x, fs, depth):
    """全长周期卷积MODWPT，db12、1/sqrt(2)归一化、Gray频率排序；统一裁边比较。"""
    wavelet = pywt.Wavelet('db12')
    filters = [np.array(wavelet.dec_lo)/np.sqrt(2), np.array(wavelet.dec_hi)/np.sqrt(2)]
    trim = (wavelet.dec_len-1)*(2**depth-1)
    start, end = max(1, trim), len(x)-max(1, trim)
    auto_rows, info_rows = [], []
    nodes = [np.fft.rfft(x)]
    for level in range(depth+1):
        auto_cells, info_cells = [], []
        for index, spectrum in enumerate(nodes):
            values = np.fft.irfft(spectrum, n=len(x))[start:end]
            se = np.abs(signal.hilbert(values))**2
            se_information = spectral_negentropy(se)
            ses_information = spectral_negentropy(np.fft.fft(se))
            low, high = index*fs/(2*len(nodes)), (index+1)*fs/(2*len(nodes))
            auto_cells.append(_cell(low, high, autocorrelation_kurtosis(se)))
            info_cells.append(_cell(low, high, (se_information+ses_information)/2,
                                    se_negentropy=se_information, ses_negentropy=ses_information))
        auto_rows.append({'level': float(level), 'cells': auto_cells})
        info_rows.append({'level': float(level), 'cells': info_cells})
        if level == depth:
            break
        responses = []
        for filt in filters:
            kernel = np.zeros(len(x))
            kernel[np.arange(len(filt))*2**level] = filt
            responses.append(np.fft.rfft(kernel))
        children = []
        for index, node in enumerate(nodes):
            order = (0, 1) if index % 2 == 0 else (1, 0)
            children.extend(node*responses[k] for k in order)
        nodes = children
    return auto_rows, info_rows, trim, end-start


def band_selection_maps(values, fs):
    x = np.asarray(values, dtype=float)
    if x.ndim != 1 or len(x) < 64 or not np.all(np.isfinite(x)) or not np.isfinite(fs) or fs <= 0:
        raise ValueError('选带需要至少64个有限单通道样本及正采样率')
    x = x-np.mean(x)
    scale = np.max(np.abs(x))
    has_energy = bool(scale > 0)
    if has_energy:
        x = x/scale
    fk_depth = max(0, min(5, int(np.floor(np.log2(len(x))))-7))
    wp_depth = 0
    # Limit live wavelet-packet spectra for long records without cutting the signal.
    while (wp_depth < 5 and len(x)-2*23*(2**(wp_depth+1)-1) >= 128
           and len(x)*2**(wp_depth+1) <= 2**22):
        wp_depth += 1
    auto, info, trim, analyzed = _modwpt_rows(x, fs, wp_depth)
    methods = {'fast_kurtogram': _finish(_fast_kurtogram(x, fs, fk_depth), has_energy),
               'autogram': _finish(auto, has_energy), 'infogram': _finish(info, has_energy)}
    return {'implementation': 'independent_band_maps_v1', 'methods': methods,
            'fast_kurtogram': {'max_level': fk_depth, 'filter_orders': [16, 24], 'transient_policy': '逐节点累计FIR起始支持裁剪',
                               'statistic': 'mean(|z-mean(z)|^4)/mean(|z-mean(z)|^2)^2-2；实根节点减3；图上截负值为0'},
            'modwpt': {'wavelet': 'db12', 'max_level': wp_depth, 'boundary': 'periodic', 'edge_trim_each_side': max(1, trim),
                       'analyzed_samples': analyzed, 'frequency_order': 'Gray', 'normalization': '1/sqrt(2)'},
            'autogram': {'variant': 'original', 'autocorrelation': 'unbiased, uncentered squared envelope',
                         'lag_fraction': [.01, .55], 'statistic': '减AC最小值后mean(AC^4)/mean(AC^2)^2，不减3'},
            'infogram': {'filterbank': 'db12 MODWPT，与Autogram共用；非作者原始滤波树的逐点复刻',
                         'statistic': 'I=mean(r*ln(r)), r=|v|^2/mean(|v|^2); v=SE或FFT(SE); (I_SE+I_SES)/2',
                         'ses_dc': 'included', 'comparison': '所有节点统一裁边、相同长度，不补零、SE不去均值'},
            'interpretation': '三种统计不跨方法比较数值；最大值只是候选，不是故障证据；未自动计算诊断包络谱'}
