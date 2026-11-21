---
name: diagnosing-bearing-faults-adaptive
description: 对滚动轴承振动波形做时域、频域、时频分析或故障诊断。用于用户提供波形或样本编号后要求分析信号、辨识内圈、外圈或滚动体故障，或复核既有诊断证据。
---

# 滚动轴承振动分析与故障诊断

## 选择任务

- **单域分析**使用 `--domain time`、`--domain frequency` 或 `--domain time_frequency`。需要多域结果时传入多个域。交付测量、图和信号状态说明。详见[信号分析输出](references/signal-analysis-output.md)。
- **完整诊断**默认稳定模式。对原始波形执行质量检查、估速、包络超额峭度选带、平方包络谱、倍频匹配及故障归属核验。只有用户明确要求精细或自适应诊断时才加 `--adaptive`。详见[自适应规则](references/adaptive-rules.md)。
- **复核**读取同一样本的结果和执行记录，逐项核对已记录的工况、用到的测量及步骤引用。

## 执行

本 Skill 不附带数据。输入为用户提供的样本清单或单通道 NPY。直接输入波形时提供 `--signal`、`--fs` 和 `--unit g` 或 `--unit a.u.`。完整诊断同时需要 `--rpm` 和轴承几何或特征频率比 JSON，经 `--geometry` 传入。部署时复制整份 Skill 目录，用 Python 3.10 或以上版本从根目录运行：

```powershell
python -m pip install -r scripts/requirements.txt
python scripts/diagnose.py run --offline --signal input.npy --fs 64000 --rpm 1200 --unit g --geometry geometry.json --output demo-evidence
python scripts/diagnose.py validate demo-evidence/result.json
python scripts/diagnose.py run --offline --signal input.npy --fs 64000 --unit g --domain time --output demo-time
```

运行前准备自己的 `input.npy` 和 `geometry.json`，并将上述采样率、转速和单位替换为实际值。每次指定新的输出目录。模型配置由环境变量或 `--env-file` 提供。

- **由脚本调用模型**时，使用 `--provider qwen`、`--provider deepseek` 或 `--provider openai` 替换 `--offline`。脚本依次执行诊断、独立评审、说明修订和引用校验。
- **使用平台模型**时，先用 `--offline` 提取证据，再按[专家判读规则](references/expert-diagnosis.md)读取 `result.json` 中的 `evidence_chain`。平台支持独立模型调用时，按[评审规则](references/expert-review.md)复核，并由诊断专家修订说明。修订保持初稿的类别。

## 交付

完成诊断后按【分析条件】【整体表现】【故障特征证据】【综合判断】报告结论，并提供本次判读波形的时域图、幅值频谱图、STFT 时频图、多频带包络谱和 `result.json`。证据链包含工具步骤、特征、频率匹配及最终说明。结论待复核时标记其状态。模型调用或报告协议失败时，交付已计算的证据、图和执行状态。

运行 `python scripts/self_check.py` 检查独立部署。
