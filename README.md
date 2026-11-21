# 滚动轴承振动分析与故障诊断 Skill

可独立部署的 Agent Skill，入口为根目录的 [SKILL.md](SKILL.md)。包含 Python 运行包、诊断与评审规则，以及 10 条 BJTU 波形示例；无需安装原始智能体项目。

## 安装

运行环境：Python 3.10+，可执行 Python 命令、安装依赖并读写本地文件。Windows、Linux 和 macOS 可按以下命令安装；Windows 已完成独立部署自检。

```bash
git clone https://github.com/liuqianli6000-ship-it/diagnosing-bearing-faults-adaptive.git
cd diagnosing-bearing-faults-adaptive
python -m venv .venv
```

Linux/macOS 激活环境：`source .venv/bin/activate`；Windows PowerShell：`.\.venv\Scripts\Activate.ps1`。

```bash
python -m pip install -r scripts/requirements.txt
python scripts/self_check.py
```

## 部署到智能体平台

将整个 `diagnosing-bearing-faults-adaptive` 目录放入目标平台指定的 Skill 目录，或将本仓库地址交给支持 GitHub Skill 导入的平台。若导入器支持选择子目录，选择仓库根目录。平台需要读取 `SKILL.md`，并保留 `scripts/`、`references/`、`assets/`、`examples/` 及根目录 JSON 文件。

支持 ZIP 导入的平台可在 GitHub 页面选择 **Code → Download ZIP**，解压后取含有 `SKILL.md` 的目录；若平台要求 ZIP 顶层即为 Skill 文件夹，按其要求重新打包。仅支持文本提示词、不能执行 Python 的平台无法运行此 Skill 的信号分析。

使用平台模型时，先运行离线命令提取证据，再让平台模型按 `SKILL.md` 和 `references/expert-diagnosis.md` 判读。平台提供独立评审能力时，再按 `references/expert-review.md` 复核。离线脚本本身不生成模型诊断结论。

## 运行示例

以下命令均从 Skill 根目录执行。每次运行使用新的输出目录。

```bash
python scripts/diagnose.py run --offline --manifest examples/metadata.jsonl --index 1 --output outputs/sample-2
python scripts/diagnose.py validate outputs/sample-2/result.json
python scripts/diagnose.py run --offline --manifest examples/metadata.jsonl --index 9 --domain time --output outputs/sample-10-time
```

索引为 0—9，对应 BJTU 000001—000010。示例用于演示流程；清单只包含输入参数，不提供诊断标签。`--domain` 可选 `time`、`frequency`、`time_frequency`，也可同时传入多个域。

完整取证生成时域、幅值频谱、STFT 时频及多频带包络图，并在输出目录保存 `result.json`。离线完整取证的结论字段为空，由平台模型依据本次证据进行判读。

## 由脚本调用模型

复制 `.env.example` 为 `.env`，填入选用服务的 API Key 和模型名，再显式传入 `--provider` 和 `--env-file`：

```bash
python scripts/diagnose.py run --provider qwen --env-file .env --manifest examples/metadata.jsonl --index 1 --output outputs/sample-2-qwen
```

支持 `qwen`、`deepseek`、`openai`。模型调用依次执行专家判读、独立评审和说明修订。`.env` 已被 Git 忽略；平台模型直接判读离线证据时无需另配脚本的模型密钥。脚本默认稳定模式，自适应处理使用 `--adaptive`。

## 分析自己的数据

单通道实数 NPY 输入，采样率单位 Hz、转速单位 rpm；幅值单位用 `g` 或未标定的 `a.u.`。完整诊断同时需要轴承几何或特征频率比 JSON。

```bash
python scripts/diagnose.py run --offline --signal input.npy --fs 64000 --rpm 1200 --unit g --geometry geometry.json --output outputs/my-signal
```

`geometry.json` 可使用以下结构，数值必须替换为目标轴承的实际参数：

```json
{"feature_frequency_ratios":{"BPFI":7.889,"BPFO":5.111,"BSF":2.203,"FTF":0.393}}
```

## 运行包与验证范围

`assets/runtime.zip` 内含可查看的 Python 源码，运行时按 `runtime_manifest.json` 核验 SHA-256 并解压到临时目录。不要单独替换运行包中的文件。维护源码位于整合项目，发布时需同步重建运行包与清单。

`self_check.py` 检查隔离部署、稳定/自适应离线取证、出图、低质量输入和运行包损坏拒绝。它不验证模型服务或诊断准确率。当前输入面向稳态滚动轴承；不同平台的实际 Skill 导入方式以其配置为准。

第三方方法说明与许可保留在 `references/` 中。
