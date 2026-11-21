"""MOMEDA —— 多点最优最小熵反卷积（Multipoint Optimal Minimum Entropy
Deconvolution Adjusted）。

当前实现映射
--------
role prompt 候选方法目录的 "Deconvolution-based method" 组（MED / MCKD / MOMEDA），
由 Apv 经 候选方法目录 工具 1 选取、工具 2 组装进分析流。候选方法目录 对该组的说明：
低信噪比下分离故障相关成分、结果依赖超参数、不满意可重跑、计算效率低。

方法名逐词对照实现
------------------
  * **Multipoint（多点）**：目标（target）不是单个冲击，而是**每隔故障周期 T
    出现一个冲击的多点序列**——这与轴承故障的物理形态一致（每个滚动体经过
    缺陷就产生一次冲击）。
  * **Optimal（最优）**：不解非线性目标做迭代，而是直接给出**闭式解**
    f = (X0ᵀX0 + εI)⁻¹ X0ᵀt（见本文件 `f = np.linalg.solve(...)`）。
  * **Adjusted**：对目标向量的构造做了调整，使解对冲击位置无偏。
  * **Minimum Entropy（最小熵）**：与 MED 同源思想——输出越稀疏（熵越小、
    冲击越突出）越好，此处用"输出与多点目标 t 的内积最大化"作为其可解代理。

与 MED/MCKD 的两点关键差别
--------------------------
1. **必须有 fault_frequency**：目标向量 t 的脉冲位置由故障周期
   T = round(fs / FCF_target) 决定，T 是目标定义的组成部分，缺 FCF 则方法
   不可定义 → 直接抛 ValueError。这与 MCKD 相同，而与 MED 不同。
2. **无迭代、无随机数**：闭式解一步得出，因此确定性是构造性保证的（MED/MCKD
   靠"固定初值 + 有限步迭代"才有确定性）。这也使 MOMEDA 是本组中计算最快的。

本实现明确固定的数值参数
------------------------------------------------------------------------
  * 滤波器长度 `filter_length = 500`；注意同组的 MED/MCKD 取 30，三者的长度
    取值当前实现均未给出，本工程按计算稳定性与资源开销折中选定（MOMEDA 的矩阵解需要更长的滤波器才
    有足够的自由度，故取值更大）。
  * 目标脉冲的半宽 `window = 3` → 每个冲击位置实际是宽 2*3+1 = 7 个采样点的
    矩形窗（比单点脉冲更宽容 T 的取整误差），脉冲起点 start = l//2。
  * 设计矩阵取全长 N-l+1 行（第 i 行为 x[i:i+l]），有效滤波器长
    l = min(filter_length, x.size//4)（限制自由度以避免病态解）。滤波按设计
    时的相关方向应用（卷积时翻转 f），否则等于用了时间反转的滤波器。
  * Tikhonov 正则化 eps = 1e-6 * trace(X0ᵀX0)/l 加在对角线上，防止 X0ᵀX0 奇异
    （噪声信号的相关矩阵常接近奇异）。
  * 输出除以 L2 范数做幅值归一化（统一各方法输出刻度）。
  以上当前实现均未给出，属本工程按上述取值选定（协议选择）。

失败边界
--------
设计矩阵每行必须完整包含 ``l`` 个样本，严格长度条件为
``x.size >= 2*l + 2*T - 1``。不满足时显式失败，不截短末行，也不把原始信号
冒充 MOMEDA 输出。``T < 2`` 同样拒绝，以避免目标向量退化。

原英文说明：MOMEDA — Multipoint Optimal Minimum Entropy Deconvolution Adjusted.
Closed-form filter solution against a multipoint target t with impulses at the
fault period T = round(fs / fault_frequency) (McDonald & Zhao 2017). Deterministic.
"""
from __future__ import annotations

import numpy as np

from ..errors import MethodNotImplementedError


def momeda(x: np.ndarray, fs: float, fault_frequency: float,
           filter_length: int = 500, window: int = 3) -> np.ndarray:
    """MOMEDA 去噪：针对多点目标序列求闭式最优 FIR 滤波器。

    步骤：① 由 FCF 定出故障周期 T；② 构造"每隔 T 一个矩形脉冲"的目标向量 t；
    ③ 把信号排成延迟矩阵 X0（每行是一个延迟版本的信号片段）；
    ④ 解正则化最小二乘 f = (X0ᵀX0 + εI)⁻¹ X0ᵀt；⑤ 卷积并 L2 归一化。
    全程无迭代、无随机数 → 确定性由构造保证。

    对应流程：候选方法目录 工具 2 中的一步去噪算子，输出随后送到 候选方法目录的
    特征提取方法。无法构造完整设计矩阵时显式失败。
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    if fault_frequency is None or fault_frequency <= 0:
        raise ValueError("MOMEDA requires the target fault frequency")
    t_period = int(round(fs / fault_frequency))   # 故障周期 T（采样点数）
    if t_period < 2:
        raise MethodNotImplementedError("MOMEDA fault period must be at least 2 samples")
    l = min(filter_length, x.size // 4)   # 有效滤波器长：限制自由度，避免病态解
    if l < 2 or x.size < 2 * l + 2 * t_period - 1:
        raise MethodNotImplementedError(
            "MOMEDA signal is too short for a complete design matrix")
    scale = float(np.max(np.abs(x)))
    if not np.all(np.isfinite(x)) or not np.isfinite(scale) or scale <= 0:
        raise ValueError("MOMEDA requires finite, nonzero signal")
    # Both Gram and its trace-based regularizer scale quadratically; the final
    # unit-norm output is unchanged by removing the input amplitude scale here.
    x = x / scale
    # 设计矩阵 X0：第 i 行为 x[i : i+l]，覆盖全部 N-l+1 个完整窗口
    x0 = np.lib.stride_tricks.sliding_window_view(x, l)
    n_target = x0.shape[0]
    # 目标向量：自 start 起每隔 T 放一个宽 2*window+1 的矩形脉冲
    tvec = np.zeros(n_target)
    start = l // 2
    positions = list(range(start, n_target, t_period))
    for p in positions:
        lo, hi = max(0, p - window), min(n_target, p + window + 1)
        tvec[lo:hi] += 1.0   # 矩形窗（比单点脉冲更宽容 T 的取整误差）
    gram = x0.T @ x0
    eps = 1e-6 * np.trace(gram) / l   # Tikhonov 正则项，防 X0ᵀX0 奇异
    f = np.linalg.solve(gram + eps * np.eye(l), x0.T @ tvec)   # 闭式最优解
    y = np.convolve(x, f[::-1], mode="same")   # 等价于设计时的相关 y[i]=Σx[i+k]f[k]
    norm = np.sqrt(np.sum(y ** 2))
    if not np.isfinite(norm) or norm <= 0:
        raise MethodNotImplementedError("MOMEDA produced an invalid zero-energy signal")
    return y / norm    # L2 归一化：统一各方法输出幅值刻度
