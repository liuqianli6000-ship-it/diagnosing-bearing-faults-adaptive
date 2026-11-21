# 三层工具目录

## L1

| 工具 | 作用 | 关键参数 |
|---|---|---|
| detrend | 去趋势 | order 0/1/2 |
| outlier_remove | MAD 野值候选 | threshold 2–12 |
| normalize | 幅值归一化 | mode zscore/rms/none |
| bandpass | 零相位带通 | low/high/order |
| assess_time_domain | 时域状态：幅值/冲击/平稳/调制 | 无 |
| assess_frequency_domain | 频域状态：主峰/SNR/窄带/谱熵 | n_peaks 3–20 |
| assess_time_frequency | 时频状态：非平稳/瞬态/谱峭度带 | n_bands 1–8 |

评估工具只测量状态，不输出 fault_type/confidence。

## L2

| 工具 | 作用 | 关键参数 |
|---|---|---|
| fft_spectrum | FFT 主峰 | 无 |
| envelope_spectrum | 包络谱 | 解调带、mode |
| spectral_kurtosis | 谱峭度带 | min_band/n_bands/n_edges |
| time_stats | 时域统计 | 无 |
| fault_frequency | 理论故障频率 | 无 |
| harmonic_match | 谐波匹配 | max_harmonics/tolerance |
| emd_decompose | EMD 选 IMF | max_imf/max_samples |
| wavelet_decompose | 小波选系数 | wavelet/level/max_samples |

## L3

| 工具 | 作用 | 关键参数 |
|---|---|---|
| adaptive_notch | 自适应陷波 | max_notches/Q |
| med | 最小熵反卷积 | filter_length/max_iter |
| mckd | 最大相关峭度反卷积 | filter_length/m/iter |
| momeda | 多点最优反卷积 | filter_length/window |
| iterative_sk | 迭代谱峭度 | band_count |
| subband_analysis | 子带分析 | band_count/overlap |
