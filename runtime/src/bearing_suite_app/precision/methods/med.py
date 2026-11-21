"""MED —— 最小熵反卷积（Minimum Entropy Deconvolution）。

当前实现映射
--------
role prompt 候选方法目录的 "Deconvolution-based method" 组，该组共 3 个方法
MED / MCKD / MOMEDA，由 Apv 智能体经 候选方法目录 工具 1 选取、工具 2 组装
进分析流。候选方法目录 对本组的定性描述为：
  Advantages:  In low SNR conditions, the deconvolution-based method can
               separate fault-related components from the signal.
  Limitations: 去噪结果依赖滤波器系数等超参数；效果不满意时可重跑以改善
               （"Running the method again may improve the denoising
               outcome if the initial result is not satisfactory"）；
               计算效率低。

原理一句话
----------
轴承故障冲击先被传递路径"卷积"抹平，MED 就是求一个有限长 FIR 滤波器把这段
卷积反向抵消回去：以 **输出信号的峭度最大化** 作为"熵最小"的可计算代理
（熵最小 ⇔ 输出最稀疏 ⇔ 冲击被凸显），因此 MED 输出脉冲性越强，故障冲击越
突出。

本方法的特殊性（与 MCKD/MOMEDA 的差别）
--------------------------------------
MED 是**无监督盲反卷积**，数学上不需要故障周期先验，只用峭度作目标；
MCKD/MOMEDA 才需要目标周期 T。因此本模块虽与另外两个方法保持统一签名
（便于 pipeline_executor.DENOISERS 用同一方式调用），但 `fault_frequency`
在本函数体内**未被使用**——这是有意的，也是与 MCKD/MOMEDA 的真实差别。

本实现明确固定的数值参数
------------------------------------------------------------------------
  * 滤波器长度 `filter_length=30`、最大迭代 `max_iter=300`；
  * 初值 f0 = 中心位置为 1 的 Kronecker delta（恒等滤波器，保证从"不去卷积"
    出发，避免随机初值引入不确定性）；
  * 优化器 L-BFGS-B、收敛容差 ftol=1e-9；
  * 输出最后除以 L2 范数做幅值归一化。
  以上当前实现均未给出，属本工程按上述取值选定（协议选择）。

确定性：固定初值 + 无随机数，同一输入两次调用结果按位相同（本工程 implementation constraints
要求所有随机过程设定种子；MED 本身无随机性）。

原英文说明：MED — Minimum Entropy Deconvolution. Maximizes the kurtosis of the
filtered signal via L-BFGS over the filter coefficients (deterministic: fixed
initialisation, no RNG).
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from ..errors import MethodNotImplementedError


def med(x: np.ndarray, fs: float, fault_frequency: float = None,
        filter_length: int = 30, max_iter: int = 300) -> np.ndarray:
    """MED 去噪：以峭度最大化为目标优化 FIR 逆滤波器并作用于原信号。

    参数中的 `fault_frequency` 为保持与 MCKD/MOMEDA 统一签名的占位参数，
    **本方法不使用它**（MED 不需要故障周期先验，见模块头部说明）；
    `fs` 同理仅用于接口统一。返回值与输入等长。

    对应流程：候选方法目录 工具 2「构造分析流」中的一步去噪算子，
    输出随后交给 候选方法目录的特征提取方法（Envelope Spectrum /
    1.5D Energy Spectrum）做频率特征提取。
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    if x.size < filter_length:
        raise MethodNotImplementedError(
            "MED signal is shorter than the selected filter length")
    scale = float(np.max(np.abs(x)))
    if not np.all(np.isfinite(x)) or not np.isfinite(scale) or scale <= 0:
        raise ValueError("MED requires finite, nonzero signal")
    # The kurtosis objective and unit-norm output are scale invariant.  Scale
    # first so finite inputs cannot overflow/underflow the fourth moments.
    x = x / scale

    def objective(f: np.ndarray) -> float:
        """返回当前 FIR 滤波器的负峭度，供最小化器求最大值。"""
        # 'same' 模式保证输出与输入等长；目标是最大化峭度，故返回其相反数
        y = np.convolve(x, f, mode="same")
        std = float(np.std(y))
        if std == 0.0:
            return 0.0
        kurt = float(np.mean((y / std) ** 4))   # 与四阶矩 / σ^4 等价，先约去幅值尺度
        return -kurt

    f0 = np.zeros(filter_length)
    f0[filter_length // 2] = 1.0    # 初值 = 中心脉冲（恒等滤波器），确定性初值
    res = minimize(objective, f0, method="L-BFGS-B",
                   options={"maxiter": max_iter, "ftol": 1e-9})
    if not np.isfinite(res.x).all():
        raise MethodNotImplementedError(f"MED optimizer produced non-finite filter: {res.message}; iterations={res.nit}")
    # v2: a budget-limited L-BFGS result is still usable.  The reference
    # implementation rejected it; for adaptive tuning we keep the best finite
    # filter and let the trial objective decide whether it helped.
    y = np.convolve(x, res.x, mode="same")
    norm = np.sqrt(np.sum(y ** 2))
    if not np.isfinite(norm) or norm <= 0:
        raise MethodNotImplementedError("MED produced an invalid zero-energy signal")
    return y / norm    # L2 归一化：统一各方法输出幅值刻度
