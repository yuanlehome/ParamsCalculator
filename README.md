# ParamsCalculator

基于 Streamlit 的大模型参数量与显存估算工具，支持主流系列（DeepSeek、Qwen、ERNIE、GLM、GPT-OSS），包含 Dense/MoE 的详细公式推导与枚举校验，并提供本地缓存与下载脚本。

## 功能亮点

- 参数量计算
  - Dense：LLaMA-like 公式（Attention=4×H²，MLP=3×H×I，Norm=RMS/LN）
  - MoE：Gate 与专家分项，支持分段配置（列表型专家数/中间维度、层区间与频率），并集计数避免层数重复
  - 激活参数与稀疏率计算（MoE TopK）
- 显存估算
  - 权重显存（FP16/BF16、FP8/Int8、FP4/Int4）
  - KV Cache 显存（优先使用 `head_dim/v_head_dim/qk_nope_head_dim/qk_rope_head_dim`，否则回退 `hidden_size/num_attention_heads`）
- 校验对比
  - “公式总计 vs 实际枚举总计”并列展示，超过 1% 给出提示
  - 无法枚举时降级为公式总计并标注 `enumeration_available=false`
- 缓存与容错
  - 使用 SQLite（`datasets/config_cache.db`）缓存 `AutoConfig.to_dict()`
  - 实例化失败时自动重拉远程配置并重试；仍失败则仅显示公式总计
  - 缺失字段补齐（如 `use_moe/use_rmsnorm/use_bias`）

## 目录结构

```
params_calculator/
  common.py        # format_number、get_dtype_size、ALIASES（字段别名映射）
  memory.py        # estimate_vram、estimate_kv_cache
  analysis.py      # 解析、公式、枚举与缓存
app.py             # Streamlit UI 入口
scripts/
  download_configs.sh   # 按模型 ID 下载 config.json 到 datasets/<org>/<model>/
requirements.txt
.pre-commit-config.yaml
```

## 快速开始

```bash
pip install -r requirements.txt
streamlit run app.py
```

UI 侧边栏输入 ModelScope 模型 ID（如 `Qwen/Qwen3-235B-A22B-Instruct-2507-FP8`），选择参数（上下文长度、Batch、TP、KV dtype 等），查看各页签：概览、详细公式、参数明细、可视化。

## 支持系列与示例

- DeepSeek：`deepseek-ai/DeepSeek-V3`
- Qwen/Qwen3：Dense 与 MoE（A3B/FP8 等）
- ERNIE：`ERNIE-4.5` 与 `ERNIE-4.5-VL-28B-A3B-Thinking`（多段 MoE）
- GLM：`ZhipuAI/GLM-4.x`
- GPT-OSS：开源镜像系列

## 公式说明（摘要）

- Dense
  - Embedding：`V × H`
  - Attention：`4 × H²`
  - MLP：`3 × H × I`
  - Norm：`RMS: 2×H` / `LayerNorm: 4×H`
  - LM Head：`H × V`（若 `tie_word_embeddings=true` 则不计）
- MoE（每层）
  - Gate：`H × (#experts_union)`
  - Experts：分段求和 `Σ (3 × H × E_i) × N_i`
  - 激活：`Gate + Σ(每专家参数) × TopK`
  - 层总计：MoE 与 Dense 分别累积；分段层使用并集计数避免重叠

## KV Cache 估算

- 单头维度优先级：`head_dim` → `v_head_dim` → `qk_nope_head_dim` → `qk_rope_head_dim` → `hidden_size // num_attention_heads`
- 头数采用 `num_key_value_heads`（MQA 下 KV 头更少）
- 公式：`layers × kv_heads × head_dim × context × 2(K+V) × batch × dtype_size ÷ tp`

## 缓存与枚举

- SQLite 缓存：`datasets/config_cache.db`
  - `configs(model_id, model_type, config_json, updated_at)`
- 实例化重试：
  1) 使用缓存/重建配置尝试实例化（`init_empty_weights` → 直接构造）
  2) 失败则远程 `AutoConfig.from_pretrained` 重拉并补齐字段后重试
  3) 仍失败则仅显示“公式总计”，`enumeration_available=false`
- 总计优先级：枚举优先，失败时回退公式总计

## 下载脚本（可选）

```bash
chmod +x scripts/download_configs.sh
./scripts/download_configs.sh
```

按模型 ID 下载 `config.json` 至 `datasets/<org>/<model>/`，用于离线检查或对照。

## 开发与规范

- 预提交钩子
  - 安装：`pre-commit install`
  - 运行：`pre-commit run --all-files`
  - 当前启用：`black`（格式化）、`ruff`（静态检查与自动修复）
- 重要说明
  - DataFrame 展示统一使用 `width` 参数（如 `width=1000`），并强制列类型为字符串，避免 PyArrow ArrowTypeError
  - 详细公式页采用结构化分段解析与代码块渲染，避免 HTML/CSS 导致错乱

## 常见问题

- “每个头的维度”显示异常
  - 现在优先读取模型配置的显式维度字段；未提供时回退按总注意力头数分配
- “公式总计 vs 枚举总计”差异大
  - 检查是否有分段层重叠（已用并集计数修复）或特殊实现（共享权重、无偏置等）
- ERNIE 等模型实例化失败（缺失字段）
  - 自动补齐常用缺失字段并重拉远程配置重试；仍失败则仅显示公式总计

## 许可证

Apache License 2.0
