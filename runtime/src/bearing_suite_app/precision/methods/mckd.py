"""MCKD —— 最大相关峭度反卷积（Maximum Correlated Kurtosis Deconvolution）。

当前实现映射
--------
role prompt 候选方法目录的 "Deconvolution-based method" 组（MED / MCKD / MOMEDA），
由 Apv 经 候选方法目录 工具 1 选取、工具 2 组装进分析流。候选方法目录 对该组的说明：
  Advantages:  低信噪比下可把故障相关成分从信号中分离出来；
  Limitations: 结果依赖滤波器系数等超参数；不满意时可重跑（"Running the
               method again may improve the denoising outcome"）；计算效率低。
一个典型组合是 MCKD 去噪后接 Envelope Spectrum：
using frequency-domain evidence）给出了本方法被真正使用的一次记录：
  "Processing Method 1: Using **maximum correlation kurtosis deconvolution**
   denoising method to denoise the signal, and then using **envelope spectrum**
   feature extraction method on the denoised signal."
其输出特征即 "1x Rolling Element Fault"、"2x Rolling Element Fault (Harmonic)"
等——正是本工程 pipeline_executor 组装的 "MCKD + Envelope Spectrum" 管线形态，
也印证了 _selection.py 把**谐波**一并计入分量评分的做法。

为什么要先验的目标故障特征频率（本方法的关键点）
----------------------------------------------
"相关峭度" CK_M(T) 度量的是**间隔为 T 的冲击串**彼此之间的相关性，而不是
普通峭度只看单个冲击的幅值稀疏性：它在目标函数里显式含有周期 T，因此
**T 不是可选输入，而是目标定义的一部分**——没有 T，CK_M(T) 无从计算。
本工程取 T = round(fs / FCF_target)，FCF_target 为**待验证位置的理论故障特征
频率**（由轴承几何参数与转速推出）。当前配置未给出 T 的取法（见
依赖边界说明），属本工程的协议选择。

该选择带来一个需要留意的语义后果：方法事实上是在"假设该位置有故障"的前提下
做增强，即假设检验式（hypothesis-confirming）的用法。这与 候选方法目录 中 Apv 的
任务 "determine if a fault has occurred at the location being analyzed" 一致，
但也意味着单次结果不足以定论，需按 候选方法目录（频谱分析）与 候选方法目录（方法
有效性评估）做交叉把关——这也是 why 本模块把"避免重复选用表现差的方法"写进
候选方法目录 两阶段机制的原因。

本实现明确固定的数值参数
------------------------------------------------------------------------
  * 相关阶数 m = 1（即 CK_1(T)：只用相邻一个周期的冲击对做相关）；
  * 滤波器长度 filter_length = 30、最大迭代 max_iter = 100；
  * 初值 f0 = 中心位置为 1 的 Kronecker delta（恒等滤波器，确定性初值）；
  * 优化器 L-BFGS-B、收敛容差 ftol = 1e-9；
  * 输出除以 L2 范数做幅值归一化（统一各方法输出刻度）；
  * T 的下界取 max(2, ...)，避免 T=0/1 使相关项退化失效；
  * 信号长度不足（< filter_length + m*T）时明确失败。
  以上当前实现均未给出，属本工程按上述取值选定（协议选择）。

确定性：固定初值 + 无随机数，同一输入两次调用结果按位相同（implementation constraints）。

原英文说明：MCKD — Maximum Correlated Kurtosis Deconvolution. Maximizes the
correlated kurtosis CK_M(T) over the filter coefficients with L-BFGS
(deterministic). T = round(fs / fault_frequency) — the target fault period of the
component under verification.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from ..errors import MethodNotImplementedError


def correlated_kurtosis(y: np.ndarray, t: int, m: int = 1) -> float:
    """相关峭度 CK_M(T)：度量间隔为 T 个采样点的冲击串的相关性。

    直觉：普通峭度只看"有没有冲击"，相关峭度额外要求这些冲击**每隔 T 个采样点
    重复出现**，因此对周期性故障冲击更有选择性，而不会被偶发外部冲击（方法评估约束中的
    role prompt 提示 "kurtosis is also susceptible to external impacts"）误导。

    CK_M(T) = Σ_n (Π_{k=0..M} y[n−kT])² / (Σy²)^(M+1)。
    所有延迟使用共同有效区间，不额外累加零延迟能量或低阶项；目标对幅值缩放不变。
    """
    y = np.asarray(y, dtype=np.float64).ravel()
    if (not isinstance(t, (int, np.integer)) or t < 1
            or not isinstance(m, (int, np.integer)) or m < 1 or y.size <= m * t):
        raise ValueError("CK requires positive integer delay/order and enough samples")
    if not np.all(np.isfinite(y)):
        raise ValueError("CK requires finite samples")
    scale = float(np.max(np.abs(y)))
    if scale == 0.0:
        return 0.0
    # CK is homogeneous of degree zero.  A unit-energy y makes its denominator
    # one without raising the original signal energy to a potentially huge power.
    y = y / scale
    y = y / np.sqrt(np.sum(y ** 2))
    product = np.ones(y.size - m * t)
    for k in range(m + 1):
        start = (m - k) * t
        product *= y[start:start + product.size]
    return float(np.sum(product ** 2))


def mckd(x: np.ndarray, fs: float, fault_frequency: float,
         m: int = 1, filter_length: int = 30, max_iter: int = 100) -> np.ndarray:
    """MCKD 去噪：以相关峭度 CK_M(T) 最大化为目标优化 FIR 滤波器。

    与 MED 的差别在于**必须有 fault_frequency**：T = round(fs / FCF_target) 是
    目标函数的固有参数（见模块头部说明），故本方法在缺 FCF 时直接抛
    ValueError，而不是退回无监督去噪——这是与 候选方法目录 工具 2「按所选方法
    构造分析流」配套的显式失败，避免产生"看起来成功但周期假设缺失"的结果。

    对应流程：候选方法目录 工具 2 中的一步去噪算子，输出随后送到 候选方法目录的
    特征提取方法；例如 MCKD 可以与 Envelope Spectrum 组合。
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    if fault_frequency is None or fault_frequency <= 0:
        raise ValueError("MCKD requires the target fault frequency (period T)")
    t = max(2, int(round(fs / fault_frequency)))   # 目标故障周期 T（下界 2 防止相关项退化）
    if x.size < filter_length + m * t:
        raise MethodNotImplementedError(
            "MCKD signal is too short for the filter and correlation delay")
    scale = float(np.max(np.abs(x)))
    if not np.all(np.isfinite(x)) or not np.isfinite(scale) or scale <= 0:
        raise ValueError("MCKD requires finite, nonzero signal")
    x = x / scale

    def objective(f: np.ndarray) -> float:
        """返回当前 FIR 滤波器的负相关峭度，供最小化器求最大值。"""
        y = np.convolve(x, f, mode="same")
        # 取负 → 用最小化器实现最大化。CK 量级约 N^-M（XJTU 约 1e-5），低于
        # L-BFGS-B 的梯度阈值 pgtol，会在两步内"收敛"退出；乘 N^M 归到 O(1)。
        return -correlated_kurtosis(y, t, m) * float(x.size) ** m

    f0 = np.zeros(filter_length)
    f0[filter_length // 2] = 1.0    # 初值 = 中心脉冲（恒等滤波器），确定性初值
    res = minimize(objective, f0, method="L-BFGS-B",
                   options={"maxiter": max_iter, "ftol": 1e-9})
    # status 1 = 达到 maxiter：滤波器已改进，按迭代预算截止属正常结果；其余失败才报错
    if res.status not in (0, 1) or not np.isfinite(res.x).all():
        raise MethodNotImplementedError("MCKD optimizer did not converge")
    y = np.convolve(x, res.x, mode="same")
    norm = np.sqrt(np.sum(y ** 2))
    if not np.isfinite(norm) or norm <= 0:
        raise MethodNotImplementedError("MCKD produced an invalid zero-energy signal")
    return y / norm    # L2 归一化：统一各方法输出幅值刻度
