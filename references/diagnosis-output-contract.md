# 诊断输出契约

## 结构化结果

| 字段 | 内容 |
| --- | --- |
| `final.fault_type` | `normal`、`inner_race_fault`、`outer_race_fault`、`rolling_element_fault` 或 `unknown` |
| `final.confidence` | 0 到 1 的证据支持分值。类别为 `unknown` 时取 0 |
| `final.source` | `skill_llm` 为模型结论，`evidence_only` 为已完成取证 |
| `final.reasoning`、`final.evidence_refs`、`final.follow_up` | 模型说明、本次实际存在的证据引用、进一步分析的事项 |
| `numeric_result`、`assessment`、`round_history` | 工具候选及证据、输入与信号状态、处理轮次 |
| `narrative.diagnosis` | 诊断类别与四段正文。网页接口对应 `diagnosis` |
| `narrative.likely_fault_type` | 有依据且需要复核时附带的候选类型 |

## 四段正文

1. **【分析条件】** 交代模式、实际工况、记录有效性和观测范围。
2. **【整体表现】** 选取可解释当前信号的时域、频域或时频指标。
3. **【故障特征证据】** 对照关键频带、理论频率、实测峰位和阶次。按实际结果说明复现、冲突或缺少观测。
4. **【综合判断】** 支持明确时给出类别。存在可优先排查的候选时，先写最可能的类型，再说明需要进一步核实的证据。离线或模型未完成判断时，说明取得的数值证据。

正文以 300–500 个非空白字符为目标。单位、频带、峰位和证据 ID 均与本次审计核对。独立脚本的 `--narrative` 输出正文，`validate` 检查结构和一致性。
