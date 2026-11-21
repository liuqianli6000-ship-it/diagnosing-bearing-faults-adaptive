# 频率匹配与跨频带支持

以下描述当前实现，不将历史评测、试验阈值或已删除方法当作现行规则。
匹配参数默认取 [MFD 配置](../mfd_method.json)，工程参数取本次加载的配置。证据等级记录工具观测结果。

## 测量链

1. 转频为本次采用 rpm/60。主链先用当前完整波形和输入转速进行转频谐波拟合，通过采用条件才用估计值，未通过则沿用输入值。故障频率为转频乘 BPFO/BPFI/BSF 比。物理几何可计算比，
   FTF 可显式提供，缺失时实现由 BPFI/BPFO 比推导；FTF 不是最终故障类别。
2. 检查原始输入、谐波是否落在可靠 Nyquist 范围、df/F 是否满足配置。
3. 状态评估保留 STFT 和包络峭度指标。最终取证采用三种独立选带方法：以 [快速谱峭度图](https://doi.org/10.1016/j.ymssp.2005.12.002) 为主，Autogram 和 Infogram 为候选。两种候选使用原项目移植的 db12 MODWPT 实现，长记录按谱数组数量限制降低层级。
   每种方法按自身指标排名，保留最多 3 个满足频率边界及最小带宽的频带。同一方法内跳过交叠达到较窄带宽一半的频带。三种方法分别解调、匹配和分段核验，记录在 `numeric_result.band_methods`。证据充分指该方法通过可诊断性门，单一故障族获得中等或强支持并有同带拟合，且其结果未因分段冲突被降为未判定。主证据充分时只用主结果；主证据不足时采用充分且结论相容的候选；其余情况综合可用方法。
4. 每带计算带通后 Hilbert 平方包络谱、Hann 窗，估计并减去低频谱底，允许负值。
   匹配实际局部峰，不把斜坡或理论标线当峰。
5. 解调后检查频带能量比、包络峰噪比和有效频带数，完成完整可诊断性门。

## 逐峰匹配

第 k 阶容差：

    T(k) = max(resolution_multiplier*df,
               k*F*sqrt(u_rpm² + u_geo² + u_slip_family²),
               minimum_tolerance_hz)

默认 resolution_multiplier=2.5，minimum_tolerance_hz=0.5 Hz。
合格峰同时满足：容差内、局部极大值、相对幅值≥0.1、局部稳健 z≥4.5。
显式覆盖后的实际值以实际加载配置为准。qualified_orders/harmonic_peaks 是测量，不是标签或概率。

## 当前支持带计数

matcher.lattice_evidence 比较故障谱列、轴频干扰谱列和可用调制谱列。
内圈调制参考为轴频，滚动体为 FTF。拟合/可见线数保留在 lattice 测量中；
拟合接受本身**不是**跨带支持的充分条件。

每族在每带满足以下任一条件才计入 evidence_bands，同一带最多计一次：

- 理论频率 1X 合格，且 lattice.best_model != "shaft"。
- 数据细化后 1X 合格，细化增益达到 decision.refined_gain_min（默认 3.0），
  且该带细化频率落在各带候选频率中位数的 ±2 df 内。增益为同一批阶次上细化与
  理论频率的平均局部 z 之差。细化范围为理论频率±5%，不等于实测 rpm。

细化只在候选频率有限、未判为干扰、偏移超过实现最小偏移时尝试。
fundamental_bands 只计理论 1X 命中，保留审计，不能代替 evidence_bands。

未基频碰撞时按支持带数定级：默认 ≥3 strong、≥2 moderate，否则 none。
碰撞按理论基频间距：两族差≤其 1X 容差之和，两族不升级为 moderate/strong；
该标志不证明已排除所有跨阶谐波重合。重叠解调带来自同一信号，等级是算法一致性要求，
不是多个独立试验或统计显著性结论。

median_residual_bins/max_residual_bins 来自 lattice 中 fault_supported 的带，
该集合不保证等同于上述支持路径所计的带；缺失残差不能当零或无故障。
lattice_models 是各带模型计数，不能据此补造实测边带。

## 确定性基线

优先 strong，无 strong 才比较 moderate。最高等级唯一候选直接选取；
多候选按 evidence_bands、合格谐波数排序。支持带数领先达到 dominance_margin
（默认 2）时选主导；否则在同级候选中比较有限的 max_residual_bins，唯一更小者主导，
不能区分则 unknown。三族 none 且完整门通过时基线为 normal。

上述结果用于数值审计。最终类别由模型按[诊断判据](diagnosis-criteria.md)综合当前证据。
