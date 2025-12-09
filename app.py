import os
import re
import time
from typing import Any, Dict, Tuple

import pandas as pd
import plotly.express as px
import streamlit as st
from modelscope import AutoConfig, AutoModel

# --- 页面配置 ---
st.set_page_config(page_title="ModelScope Model Params Calculator", page_icon="🧮", layout="wide")


# --- 核心逻辑函数 ---
def format_number(num: int) -> str:
    """将数字格式化为 B (Billion) 或 M (Million)"""
    if num >= 1_000_000_000:
        return f"{num / 1_000_000_000:.2f} B"
    elif num >= 1_000_000:
        return f"{num / 1_000_000:.2f} M"
    else:
        return f"{num:,}"


def estimate_vram(param_count: int) -> Dict[str, str]:
    """估算不同精度下的权重显存占用"""

    def bytes_to_gb(b):
        return f"{b / (1024 ** 3):.2f} GB"

    return {
        "FP32 (4 bytes)": bytes_to_gb(param_count * 4),
        "FP16/BF16 (2 bytes)": bytes_to_gb(param_count * 2),
        "Int8 (1 byte)": bytes_to_gb(param_count * 1),
        "Int4 (0.5 byte)": bytes_to_gb(param_count * 0.5),
    }


def estimate_kv_cache(model_config, context_length=2048, batch_size=1, dtype="fp16", tp=1):
    """估算 KV Cache 显存占用"""
    num_layers = getattr(model_config, "num_hidden_layers", -1)
    num_heads = getattr(model_config, "num_key_value_heads", -1)
    hidden_size = getattr(model_config, "hidden_size", -1)
    head_dim = getattr(model_config, "head_dim", -1)

    if num_heads < 0 or hidden_size < 0:
        return "0.00 GB", {}
    if head_dim < 0:
        head_dim = hidden_size // num_heads

    dtype_size = {"fp32": 4, "fp16": 2, "bf16": 2, "fp8": 1, "int8": 1, "int4": 0.5}.get(dtype.lower(), 2)

    # 计算公式步骤
    calculation_steps = {
        "层数": num_layers,
        "注意力头数": num_heads,
        "每个头的维度": head_dim,
        "上下文长度": context_length,
        "批大小": batch_size,
        "数据类型字节数": dtype_size,
        "KV向量数": 2,  # Key和Value
        "张量并行度": tp,
    }

    # 计算公式：layers × heads × head_dim × context_length × 2 (K+V) × batch_size × dtype_size / tp
    kv_cache_bytes = num_layers * num_heads * head_dim * context_length * 2 * batch_size * dtype_size
    kv_cache_bytes /= tp

    return f"{kv_cache_bytes / (1024 ** 3):.2f} GB", calculation_steps


def extract_layer_index(name: str) -> int:
    """提取层编号用于排序"""
    match = re.search(r"layers.(\d+)", name)
    return int(match.group(1)) if match else -1


def identify_param_type(name: str) -> str:
    """识别关键参数类型"""
    name_lower = name.lower()
    if "embedding" in name_lower:
        return "embedding"
    elif any(k in name_lower for k in ["q_proj", "k_proj", "v_proj", "attn", "attention"]):
        return "attention"
    elif (
        "mlp" in name_lower
        or "fc" in name_lower
        or "gate" in name_lower
        or "up_proj" in name_lower
        or "down_proj" in name_lower
    ):
        return "mlp"
    elif "norm" in name_lower or "ln" in name_lower:
        return "norm"
    elif "lm_head" in name_lower or "head" in name_lower:
        return "head"
    else:
        return "other"


def get_dtype_size(dtype: str) -> float:
    """获取数据类型对应的字节数"""
    dtype_map = {"fp32": 4, "fp16": 2, "bf16": 2, "fp8": 1, "int8": 1, "int4": 0.5}
    return dtype_map.get(dtype.lower(), 2)


def calculate_model_params_detail(config):
    """计算模型参数量的详细公式，支持Dense和MoE模型"""
    details = {}
    formulas = []

    try:
        model_type = getattr(config, "model_type", "unknown").lower()
        vocab_size = getattr(config, "vocab_size", 0)
        hidden_size = getattr(config, "hidden_size", 0)
        num_layers = getattr(config, "num_hidden_layers", 0)
        num_attention_heads = getattr(config, "num_attention_heads", 0)
        intermediate_size = getattr(config, "intermediate_size", 0)

        # MoE 相关参数
        num_experts = getattr(config, "num_local_experts", getattr(config, "num_experts", 0))
        num_experts_per_tok = getattr(config, "num_experts_per_tok", getattr(config, "top_k", 0))
        expert_intermediate_size = getattr(
            config, "expert_intermediate_size", getattr(config, "ffn_hidden_size", intermediate_size)
        )

        # 判断是否是MoE模型
        is_moe_model = num_experts > 0
        if is_moe_model:
            model_type_with_moe = f"{model_type} (MoE)"
        else:
            model_type_with_moe = f"{model_type} (Dense)"

        # 存储基础信息
        details["基础信息"] = {
            "模型类型": model_type_with_moe,
            "词表大小": vocab_size,
            "隐藏层维度": hidden_size,
            "层数": num_layers,
            "注意力头数": num_attention_heads,
            "中间层维度": intermediate_size,
            "是否MoE": "是" if is_moe_model else "否",
        }

        if is_moe_model:
            details["基础信息"]["专家数量"] = num_experts
            details["基础信息"]["每token专家数"] = num_experts_per_tok
            details["基础信息"]["专家中间层维度"] = expert_intermediate_size

        # Transformer类模型通用计算公式
        if (
            "llama" in model_type
            or "qwen" in model_type
            or "deepseek" in model_type
            or "ernie" in model_type
            or "mixtral" in model_type
        ):
            total_params = 0

            # 1. Embedding 参数
            embedding_params = vocab_size * hidden_size
            total_params += embedding_params
            formulas.append("### 1. Embedding 层参数")
            formulas.append("词表大小 × 隐藏层维度")
            formulas.append(
                f"{format_number(vocab_size)} × {format_number(hidden_size)} = **{format_number(embedding_params)}**"
            )

            # 2. Attention 参数（每层）
            head_dim = hidden_size // num_attention_heads

            # QKV投影参数
            qkv_params_per_layer = (hidden_size * hidden_size) * 3  # Q, K, V 投影矩阵
            # Output投影参数
            output_proj_params = hidden_size * hidden_size

            attention_params_per_layer = qkv_params_per_layer + output_proj_params

            formulas.append("\n### 2. Attention 层参数（每层）")
            formulas.append("#### a) QKV投影 (Q, K, V 各一个线性层)")
            formulas.append("隐藏层维度 × 隐藏层维度 × 3")
            formulas.append(
                f"{format_number(hidden_size)} × {format_number(hidden_size)} × 3 = **{format_number(qkv_params_per_layer)}**"
            )

            formulas.append("\n#### b) Output投影")
            formulas.append("隐藏层维度 × 隐藏层维度")
            formulas.append(
                f"{format_number(hidden_size)} × {format_number(hidden_size)} = **{format_number(output_proj_params)}**"
            )

            formulas.append("\n#### c) 每层Attention总参数")
            formulas.append("QKV投影 + Output投影")
            formulas.append(
                f"{format_number(qkv_params_per_layer)} + {format_number(output_proj_params)} = **{format_number(attention_params_per_layer)}**"
            )

            # 3. MLP/FFN 参数（每层）
            mlp_params_per_layer = 0
            mlp_calculation = []

            if intermediate_size > 0 or expert_intermediate_size > 0:
                if is_moe_model:
                    # MoE 模型计算
                    formulas.append("\n### 3. MoE层参数（每层）")

                    # 门控网络参数（gate或router）
                    gate_params = hidden_size * num_experts
                    mlp_params_per_layer += gate_params
                    mlp_calculation.append(
                        f"Gate网络: {format_number(hidden_size)} × {num_experts} = {format_number(gate_params)}"
                    )

                    # 每个专家：gate_proj + up_proj + down_proj
                    expert_params_per_layer = (hidden_size * expert_intermediate_size) * 2 + (
                        expert_intermediate_size * hidden_size
                    )
                    mlp_calculation.append("每个专家参数: gate_proj + up_proj + down_proj")
                    mlp_calculation.append(
                        f"  = ({format_number(hidden_size)} × {format_number(expert_intermediate_size)} × 2) + ({format_number(expert_intermediate_size)} × {format_number(hidden_size)})"
                    )
                    mlp_calculation.append(f"  = {format_number(expert_params_per_layer)}")

                    # 所有专家总参数
                    all_experts_params = expert_params_per_layer * num_experts
                    mlp_params_per_layer += all_experts_params
                    mlp_calculation.append(
                        f"所有{num_experts}个专家总参数: {format_number(expert_params_per_layer)} × {num_experts} = {format_number(all_experts_params)}"
                    )

                    formulas.extend(mlp_calculation)

                    # MoE激活参数（每个token实际激活的参数）
                    active_params_per_layer = gate_params + (expert_params_per_layer * num_experts_per_tok)
                    formulas.append("\n#### d) 每token激活的MoE参数（推理时）")
                    formulas.append(f"Gate参数 + (每个专家参数 × {num_experts_per_tok}个激活专家)")
                    formulas.append(
                        f"{format_number(gate_params)} + ({format_number(expert_params_per_layer)} × {num_experts_per_tok}) = **{format_number(active_params_per_layer)}**"
                    )

                    details["MoE信息"] = {
                        "每层总MoE参数": mlp_params_per_layer,
                        "每个专家参数": expert_params_per_layer,
                        "每token激活参数": active_params_per_layer,
                        "稀疏性": f"{(num_experts_per_tok / num_experts * 100):.1f}%",
                    }

                else:
                    # Dense架构：gate_proj + up_proj + down_proj
                    mlp_params_per_layer = (hidden_size * intermediate_size) * 2 + (intermediate_size * hidden_size)

                    formulas.append("\n### 3. MLP层参数（每层）- Llama架构")
                    formulas.append("#### a) gate_proj + up_proj")
                    formulas.append("隐藏层维度 × 中间层维度 × 2")
                    formulas.append(
                        f"{format_number(hidden_size)} × {format_number(intermediate_size)} × 2 = **{format_number(hidden_size * intermediate_size * 2)}**"
                    )

                    formulas.append("\n#### b) down_proj")
                    formulas.append("中间层维度 × 隐藏层维度")
                    formulas.append(
                        f"{format_number(intermediate_size)} × {format_number(hidden_size)} = **{format_number(intermediate_size * hidden_size)}**"
                    )

                    formulas.append("\n#### c) 每层MLP总参数")
                    formulas.append(f"{format_number(mlp_params_per_layer)}")

            # 4. LayerNorm 参数（每层）
            # 两个LayerNorm：attention之前的和MLP之前的
            # 每个LayerNorm：gamma (hidden_size) + beta (hidden_size)
            norm_params_per_layer = hidden_size * 2 * 2  # 2个LayerNorm，每个2个参数

            formulas.append("\n### 4. LayerNorm 参数（每层）")
            formulas.append("隐藏层维度 × 2（gamma和beta）× 2（pre-attention和pre-MLP）")
            formulas.append(f"{format_number(hidden_size)} × 2 × 2 = **{format_number(norm_params_per_layer)}**")

            # 5. 每层总参数
            if is_moe_model:
                # MoE模型：每层参数 = Attention + MoE + LayerNorm
                params_per_layer = attention_params_per_layer + mlp_params_per_layer + norm_params_per_layer
                active_params_per_layer = (
                    attention_params_per_layer + details["MoE信息"]["每token激活参数"] + norm_params_per_layer
                )

                formulas.append("\n### 5. 每层总参数 (MoE模型)")
                formulas.append("#### a) 总参数（包含所有专家）")
                formulas.append("Attention + MoE(所有专家) + LayerNorm")
                formulas.append(
                    f"{format_number(attention_params_per_layer)} + {format_number(mlp_params_per_layer)} + {format_number(norm_params_per_layer)} = **{format_number(params_per_layer)}**"
                )

                formulas.append("\n#### b) 激活参数（每token实际使用）")
                formulas.append("Attention + MoE(激活专家) + LayerNorm")
                formulas.append(
                    f"{format_number(attention_params_per_layer)} + {format_number(details['MoE信息']['每token激活参数'])} + {format_number(norm_params_per_layer)} = **{format_number(active_params_per_layer)}**"
                )

                # 存储MoE激活参数
                details["MoE信息"]["每层激活参数"] = active_params_per_layer
            else:
                # Dense模型
                params_per_layer = attention_params_per_layer + mlp_params_per_layer + norm_params_per_layer

                formulas.append("\n### 5. 每层总参数")
                formulas.append("Attention + MLP + LayerNorm")
                formulas.append(
                    f"{format_number(attention_params_per_layer)} + {format_number(mlp_params_per_layer)} + {format_number(norm_params_per_layer)} = **{format_number(params_per_layer)}**"
                )

            # 6. 所有层参数
            all_layers_params = params_per_layer * num_layers
            total_params += all_layers_params

            if is_moe_model:
                # MoE模型的激活参数总量
                all_active_params = active_params_per_layer * num_layers
                formulas.append(f"\n### 6. 所有{num_layers}层总参数 (MoE模型)")
                formulas.append("#### a) 总参数（包含所有专家）")
                formulas.append("每层总参数 × 层数")
                formulas.append(
                    f"{format_number(params_per_layer)} × {num_layers} = **{format_number(all_layers_params)}**"
                )

                formulas.append("\n#### b) 激活参数总量（每token实际使用）")
                formulas.append("每层激活参数 × 层数")
                formulas.append(
                    f"{format_number(active_params_per_layer)} × {num_layers} = **{format_number(all_active_params)}**"
                )

                details["MoE信息"]["总激活参数"] = all_active_params
            else:
                formulas.append(f"\n### 6. 所有{num_layers}层总参数")
                formulas.append("每层参数 × 层数")
                formulas.append(
                    f"{format_number(params_per_layer)} × {num_layers} = **{format_number(all_layers_params)}**"
                )

            # 7. 输出层 (LM Head) 参数
            lm_head_params = hidden_size * vocab_size
            total_params += lm_head_params

            formulas.append("\n### 7. 输出层 (LM Head) 参数")
            formulas.append("隐藏层维度 × 词表大小")
            formulas.append(
                f"{format_number(hidden_size)} × {format_number(vocab_size)} = **{format_number(lm_head_params)}**"
            )

            # 8. 最终总计
            if is_moe_model:
                formulas.append("\n### 8. 模型总参数量 (MoE模型)")
                formulas.append("#### a) 总参数（包含所有专家）")
                formulas.append("Embedding + 所有层(总) + LM Head")
                formulas.append(
                    f"{format_number(embedding_params)} + {format_number(all_layers_params)} + {format_number(lm_head_params)} = **{format_number(total_params)}**"
                )

                # MoE模型的激活参数总量（推理时）
                total_active_params = embedding_params + all_active_params + lm_head_params
                formulas.append("\n#### b) 激活参数总量（每token实际使用）")
                formulas.append("Embedding + 所有层(激活) + LM Head")
                formulas.append(
                    f"{format_number(embedding_params)} + {format_number(all_active_params)} + {format_number(lm_head_params)} = **{format_number(total_active_params)}**"
                )

                # 计算稀疏率和激活参数比例
                sparsity = (1 - (total_active_params / total_params)) * 100
                formulas.append("\n#### c) 稀疏率")
                formulas.append("1 - (激活参数 / 总参数)")
                formulas.append(
                    f"1 - ({format_number(total_active_params)} / {format_number(total_params)}) = **{sparsity:.1f}%**"
                )

                # 存储MoE相关计算结果
                details["详细计算"] = {
                    "Embedding参数": embedding_params,
                    "每层Attention参数": attention_params_per_layer,
                    "每层MoE总参数": mlp_params_per_layer,
                    "每层激活MoE参数": details["MoE信息"]["每token激活参数"] if is_moe_model else 0,
                    "每层LayerNorm参数": norm_params_per_layer,
                    "每层总参数": params_per_layer,
                    "每层激活参数": active_params_per_layer if is_moe_model else params_per_layer,
                    "所有层总参数": all_layers_params,
                    "所有层激活参数": all_active_params if is_moe_model else all_layers_params,
                    "LM Head参数": lm_head_params,
                    "总计（含所有专家）": total_params,
                    "总计（激活参数）": total_active_params if is_moe_model else total_params,
                    "稀疏率": f"{sparsity:.1f}%",
                }
            else:
                formulas.append("\n### 8. 模型总参数量")
                formulas.append("Embedding + 所有层 + LM Head")
                formulas.append(
                    f"{format_number(embedding_params)} + {format_number(all_layers_params)} + {format_number(lm_head_params)} = **{format_number(total_params)}**"
                )

                # 存储详细计算结果
                details["详细计算"] = {
                    "Embedding参数": embedding_params,
                    "每层Attention参数": attention_params_per_layer,
                    "每层MLP参数": mlp_params_per_layer,
                    "每层LayerNorm参数": norm_params_per_layer,
                    "每层总参数": params_per_layer,
                    "所有层总参数": all_layers_params,
                    "LM Head参数": lm_head_params,
                    "总计": total_params,
                }

            details["公式"] = formulas

        else:
            formulas.append(f"模型类型 '{model_type}' 的详细计算公式未实现")
            details["公式"] = formulas

    except Exception as e:
        formulas.append(f"计算详细公式时出错: {str(e)}")
        details["公式"] = formulas

    return details


@st.cache_data(show_spinner=False)
def analyze_model_structure(model_id: str, trust_remote_code: bool) -> Tuple[bool, Any, pd.DataFrame, str, Any]:
    """
    下载 Config 并实例化 Meta Model，统计参数。
    返回: (是否成功, 统计信息字典, 详细参数DataFrame, 错误信息, config对象)
    """
    try:
        # 设置缓存路径
        cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "modelscope", "hub")
        os.makedirs(cache_dir, exist_ok=True)

        # 从ModelScope加载配置
        try:
            config = AutoConfig.from_pretrained(model_id, trust_remote_code=trust_remote_code, cache_dir=cache_dir)
            st.success(f"✅ 成功从ModelScope加载配置: {model_id}")
        except Exception as e:
            return False, None, None, f"ModelScope配置加载失败: {str(e)}", None

        # 计算详细公式
        detail_calculation = calculate_model_params_detail(config)

        # 尝试加载模型结构（使用meta tensor）
        try:
            from accelerate import init_empty_weights

            with init_empty_weights():
                model = AutoModel.from_config(config, trust_remote_code=trust_remote_code)
        except ImportError:
            # 如果accelerate不可用，尝试直接加载但捕获错误
            st.warning("⚠️ accelerate库未安装，使用普通方式加载（可能消耗更多内存）")
            try:
                model = AutoModel.from_config(config, trust_remote_code=trust_remote_code)
            except Exception as e:
                return False, None, None, f"模型结构初始化失败（无meta tensor）: {str(e)}", None

        total_params = 0
        trainable_params = 0
        param_data = []

        for name, param in model.named_parameters():
            num_params = param.numel()
            total_params += num_params
            if param.requires_grad:
                trainable_params += num_params

            param_data.append(
                {
                    "Full Name": name,
                    "Group": name.split(".")[0] if len(name.split(".")) > 0 else "base",
                    "SubGroup": name.split(".")[1] if len(name.split(".")) > 1 else "other",
                    "Shape": str(tuple(param.shape)),
                    "Count": num_params,
                    "Dtype": str(param.dtype).replace("torch.", ""),
                    "LayerIdx": extract_layer_index(name),
                    "ParamType": identify_param_type(name),
                }
            )

        df_params = pd.DataFrame(param_data)

        # 使用详细计算中的总计，或者使用统计的总计
        calculated_total = detail_calculation.get("详细计算", {}).get("总计", 0)
        if calculated_total > 0:
            final_total = calculated_total
        else:
            final_total = total_params

        info = {
            "model_type": getattr(config, "model_type", "unknown"),
            "total_params": final_total,
            "trainable_params": trainable_params,
            "architectures": getattr(config, "architectures", ["Unknown"]),
            "vocab_size": getattr(config, "vocab_size", "N/A"),
            "hidden_size": getattr(config, "hidden_size", "N/A"),
            "num_layers": getattr(config, "num_hidden_layers", 0),
            "num_heads": getattr(config, "num_attention_heads", 0),
            "max_position_embeddings": getattr(config, "max_position_embeddings", "N/A"),
            "intermediate_size": getattr(config, "intermediate_size", "N/A"),
            "source": "ModelScope",
            "detail_calculation": detail_calculation,
        }

        return True, info, df_params, "", config
    except Exception as e:
        return False, None, None, str(e), None


# --- UI 布局 ---
st.title("🧮 ModelScope 模型参数透视镜")
st.markdown(
    """
此工具通过读取 ModelScope 模型的 `config.json` 并构建 **Meta Tensor** 来计算参数量。
**特点：** 无需下载庞大权重文件，秒级分析 70B+ 模型，节省内存，**并展示详细的参数量计算公式**。
"""
)

with st.sidebar:
    st.header("设置")

    # 模型输入
    model_input = st.text_input(
        "ModelScope 模型 ID",
        value="Qwen/Qwen3-235B-A22B-Instruct-2507-FP8",
        help="格式：组织名/模型名，如 Qwen/Qwen3-235B-A22B-Instruct-2507-FP8",
    )

    trust_remote = st.checkbox(
        "Trust Remote Code", value=True, help="大多数ModelScope模型需要此选项，否则可能无法加载配置"
    )

    st.divider()
    st.subheader("推理配置")
    # 修改上下文长度选择
    context_options = {
        "1K (1024)": 1024,
        "4K (4096)": 4096,
        "8K (8192)": 8192,
        "16K (16384)": 16384,
        "32K (32768)": 32768,
        "64K (65536)": 65536,
        "128K (131072)": 131072,
        "自定义": "custom",
    }

    context_choice = st.selectbox(
        "上下文长度", options=list(context_options.keys()), index=4, help="选择预设长度或自定义"  # 默认选择 32K
    )

    if context_choice == "自定义":
        context_length = st.number_input(
            "输入自定义上下文长度",
            value=32768,
            min_value=1,
            max_value=1_000_000,
            step=1024,
            help="输入具体的上下文长度值",
        )
    else:
        context_length = context_options[context_choice]
    batch_size = st.number_input("批大小", value=8, min_value=1, step=1, help="推理时的批量大小")
    tp = st.number_input("张量并行度 (TP)", value=2, min_value=1, step=1, help="模型并行度，通常用于多卡推理")
    dtype_select = st.selectbox(
        "KV Cache 数据类型",
        options=["fp16", "bf16", "fp32", "fp8", "int8", "int4"],
        index=0,
        help="KV Cache 存储的数据精度",
    )

# 如果有session state中的模型ID，更新输入框
if "model_input" in st.session_state:
    model_input = st.session_state.model_input

run_btn = st.button("🚀 开始分析", type="primary", width="stretch")

if run_btn and model_input:
    status_container = st.status("正在连接 ModelScope...", expanded=True)
    start_time = time.time()

    with status_container:
        st.write("📡 数据源: ModelScope")
        st.write(f"🔍 模型ID: {model_input}")
        st.write("📊 正在计算详细参数公式...")

    success, info, df, error_msg, config = analyze_model_structure(model_input, trust_remote)

    if success:
        elapsed_time = time.time() - start_time
        status_container.update(label=f"✅ 分析完成！耗时 {elapsed_time:.2f}秒", state="complete", expanded=False)

        # --- 主显示区域 ---
        tab_overview, tab_formula, tab_details, tab_viz = st.tabs(
            ["📊 概览", "🧮 详细公式", "🔍 详细参数", "📈 可视化"]
        )

        with tab_overview:
            st.subheader("📊 模型基本信息")

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("总参数量", format_number(info["total_params"]))
            col2.metric("模型架构", info["model_type"])
            col3.metric("隐藏层维度", info["hidden_size"])
            col4.metric("词表大小", info["vocab_size"])

            col5, col6, col7, col8 = st.columns(4)
            col5.metric("层数", info["num_layers"])
            col6.metric("注意力头数", info["num_heads"])
            col7.metric("中间层大小", info["intermediate_size"])
            col8.metric("最大序列长度", info["max_position_embeddings"])

            # 显示MoE信息（如果适用）
            detail_info = info.get("detail_calculation", {})
            if "MoE信息" in detail_info:
                st.subheader("🧩 MoE 模型信息")
                moe_info = detail_info["MoE信息"]

                col9, col10, col11, col12 = st.columns(4)
                if "专家数量" in moe_info:
                    col9.metric("专家数量", moe_info["专家数量"])
                if "每token专家数" in moe_info:
                    col10.metric("每token专家数", moe_info["每token专家数"])
                if "稀疏性" in moe_info:
                    col11.metric("稀疏率", moe_info["稀疏性"])
                if "总计（激活参数）" in detail_info["详细计算"]:
                    col12.metric("激活参数", format_number(detail_info["详细计算"]["总计（激活参数）"]))

            # 显示数据源
            st.info(f"📡 数据源: {info['source']}")

            # --- 权重显存 ---
            st.subheader("💾 理论显存占用 (仅权重)")
            vram_info = estimate_vram(info["total_params"])
            v_cols = st.columns(4)
            for idx, (dtype, size) in enumerate(vram_info.items()):
                v_cols[idx].info(f"**{dtype}**\n\n{size}")

            # --- KV Cache 显存 ---
            if info["num_layers"] > 0 and info["num_heads"] > 0:
                kv_size, kv_steps = estimate_kv_cache(config, context_length, batch_size, dtype_select, tp)
                st.info(
                    f"⚡ KV Cache 显存估算 ({dtype_select}, context={context_length}, batch={batch_size}, TP={tp}): {kv_size}"
                )

                with st.expander("查看KV Cache计算公式"):
                    st.write("### KV Cache 计算公式")
                    # 创建两列布局，让公式有更多空间
                    col_formula, col_explanation = st.columns([1, 1])

                    with col_formula:
                        st.markdown("**计算公式:**")
                        # 使用更简洁的 LaTeX 公式并确保正确显示
                        st.latex(r"""\text{KB} = \frac{L \times H \times D \times C \times 2 \times B \times S}{TP}""")

                    with col_explanation:
                        st.markdown("**变量说明:**")
                        st.markdown("- $L$ = 层数")
                        st.markdown("- $H$ = 注意力头数")
                        st.markdown("- $D$ = 每个头的维度")
                        st.markdown("- $C$ = 上下文长度")
                        st.markdown("- $B$ = 批大小")
                        st.markdown("- $S$ = 数据类型字节数")
                        st.markdown("- $TP$ = 张量并行度")
                        st.markdown("- $2$ = Key 和 Value 两个向量")

                    st.write("**计算步骤:**")
                    for key, value in kv_steps.items():
                        st.write(f"- {key}: {value}")

                    st.write("\n**具体计算:**")
                    st.write(
                        f"{info['num_layers']} × {info['num_heads']} × {info['hidden_size'] // info['num_heads']} × {context_length} × 2 × {batch_size} × {get_dtype_size(dtype_select)} ÷ {tp}"
                    )
                    st.write(f"= {kv_size}")
            else:
                st.warning("⚠️ 无法计算KV Cache：模型层数或注意力头数为0")

        with tab_formula:
            st.subheader("🧮 参数详细计算公式")

            detail_info = info.get("detail_calculation", {})

            # 显示基础信息
            if "基础信息" in detail_info:
                st.write("### 模型配置信息")
                base_info = detail_info["基础信息"]
                info_cols = st.columns(3)
                info_items = list(base_info.items())

                for i in range(0, len(info_items), 3):
                    for j in range(3):
                        if i + j < len(info_items):
                            key, value = info_items[i + j]
                            info_cols[j].metric(key, value)

            # 显示详细公式
            if "公式" in detail_info:
                st.write("### 参数量计算公式推导")

                # 创建一个可折叠的代码区域显示公式
                formula_text = "\n".join(detail_info["公式"])

                # 使用markdown显示公式，增强可读性
                st.markdown(
                    """
                <style>
                .formula-box {
                    background-color: #f8f9fa;
                    border-left: 4px solid #4e73df;
                    padding: 1rem;
                    margin: 1rem 0;
                    border-radius: 0.25rem;
                }
                .formula-step {
                    margin: 0.5rem 0;
                    padding: 0.5rem;
                    background-color: #ffffff;
                    border-radius: 0.25rem;
                }
                </style>
                """,
                    unsafe_allow_html=True,
                )

                # 将公式文本分割成步骤显示
                formula_lines = detail_info["公式"]
                current_section = []

                for line in formula_lines:
                    if line.startswith("### "):
                        # 显示之前的部分
                        if current_section:
                            st.markdown(
                                f'<div class="formula-box">{"<br>".join(current_section)}</div>',
                                unsafe_allow_html=True,
                            )
                            current_section = []
                        # 新的大标题
                        st.markdown(f"**{line[4:]}**")
                    elif line.startswith("#### "):
                        # 显示之前的部分
                        if current_section:
                            st.markdown(
                                f'<div class="formula-step">{"<br>".join(current_section)}</div>',
                                unsafe_allow_html=True,
                            )
                            current_section = []
                        # 小标题
                        st.markdown(f"*{line[5:]}*")
                    elif line.strip():
                        current_section.append(line)

                # 显示最后的部分
                if current_section:
                    st.markdown(
                        f'<div class="formula-step">{"<br>".join(current_section)}</div>', unsafe_allow_html=True
                    )

                # 显示Latex公式总结
                st.write("### 公式总结")
                st.latex(
                    r"""
                \begin{aligned}
                \text{总参数} &= \text{Embedding} + N \times (\text{Attention层} + \text{MLP层} + \text{LayerNorm层}) + \text{LM Head} \\
                \text{Embedding} &= V \times H \\
                \text{Attention层} &= 4 \times H^2 \\
                \text{MLP层} &= 3 \times H \times I \quad (\text{Llama架构}) \\
                \text{LayerNorm层} &= 4 \times H \\
                \text{LM Head} &= H \times V
                \end{aligned}
                """
                )

                st.write("其中：")
                st.write("- $V$ = 词表大小")
                st.write("- $H$ = 隐藏层维度")
                st.write("- $I$ = 中间层维度")
                st.write("- $N$ = 层数")

            else:
                st.warning("未能生成详细计算公式")

        with tab_details:
            st.subheader("🔍 详细参数统计")

            if not df.empty:
                # 参数统计摘要
                st.write("### 参数类型统计")
                type_stats = df.groupby("ParamType")["Count"].sum().reset_index()
                type_stats = type_stats.sort_values("Count", ascending=False)

                cols = st.columns(1)
                with cols[0]:
                    # 显示百分比
                    type_stats["Percentage"] = (type_stats["Count"] / type_stats["Count"].sum() * 100).round(2)
                    st.dataframe(type_stats[["ParamType", "Count", "Percentage"]], width="stretch")

                # 层参数统计
                st.write("### 每层参数统计")
                layer_stats = df[df["LayerIdx"] >= 0].groupby("LayerIdx")["Count"].sum().reset_index()
                layer_stats = layer_stats.sort_values("LayerIdx")

                if not layer_stats.empty:
                    # 计算平均值
                    avg_params_per_layer = layer_stats["Count"].mean()
                    st.info(f"平均每层参数: {format_number(int(avg_params_per_layer))}")

                    # 显示层参数表格
                    st.dataframe(layer_stats, width="stretch")

                # 详细参数表
                st.write("### 完整参数列表")
                st.dataframe(df[["Full Name", "Shape", "Count", "ParamType", "LayerIdx"]], width="stretch", height=500)
            else:
                st.warning("未能解析出详细参数结构。")

        with tab_viz:
            st.subheader("📈 参数可视化")

            if not df.empty:
                # 1. 参数类型分布饼图
                col1, col2 = st.columns(2)

                with col1:
                    type_df = df.groupby("ParamType")["Count"].sum().reset_index()
                    fig1 = px.pie(type_df, values="Count", names="ParamType", title="参数类型分布", hole=0.3)
                    st.plotly_chart(fig1, width="stretch")

                with col2:
                    # 2. 层级分布条形图
                    if df["LayerIdx"].max() > 0:
                        layer_df = df[df["LayerIdx"] >= 0].groupby("LayerIdx")["Count"].sum().reset_index()
                        fig2 = px.bar(
                            layer_df,
                            x="LayerIdx",
                            y="Count",
                            title="各层参数分布",
                            labels={"LayerIdx": "层索引", "Count": "参数量"},
                        )
                        st.plotly_chart(fig2, width="stretch")

                # 3. Treemap
                st.write("### 层级结构分布图")
                df_grouped = df.groupby(["Group", "LayerIdx", "SubGroup", "ParamType"])["Count"].sum().reset_index()
                df_grouped = df_grouped.sort_values(["Group", "LayerIdx"])

                fig3 = px.treemap(
                    df_grouped,
                    path=[px.Constant(model_input), "Group", "LayerIdx", "ParamType", "SubGroup"],
                    values="Count",
                    color="LayerIdx",
                    hover_data=["Count", "ParamType"],
                    title=f"{model_input} 参数层级分布",
                )
                fig3.update_traces(textinfo="label+value")
                st.plotly_chart(fig3, width="stretch")
            else:
                st.warning("没有足够的数据进行可视化。")

    else:
        status_container.update(label="❌ 出错了", state="error", expanded=True)
        st.error(f"无法加载模型信息: {error_msg}")

        if "404" in error_msg or "not found" in error_msg.lower():
            st.warning("请检查模型 ID 是否拼写正确，或者该模型是否存在。")
            st.markdown("🔍 你可以在 [ModelScope](https://modelscope.cn/models) 搜索模型")

        if "trust_remote_code" in error_msg:
            st.warning("ModelScope模型通常需要Trust Remote Code选项，请确保已勾选。")

        st.info("💡 **常见问题解决方案:**")
        st.markdown("1. 确保模型ID格式正确：`组织名/模型名`")
        st.markdown("2. 尝试勾选 **Trust Remote Code** 选项")

elif run_btn and not model_input:
    st.warning("请输入 ModelScope 模型 ID。")

# 添加页脚
st.divider()
st.caption(
    """
**💡 使用提示:**
- 本工具通过分析模型配置和结构自动计算参数量，无需下载权重文件
- 详细公式推导基于Transformer架构，对于非标准架构可能略有差异
- KV Cache计算适用于Decoder-only语言模型
"""
)

# 添加ModelScope链接
st.markdown("---")
st.markdown(
    "🔗 [ModelScope 官网](https://modelscope.cn) | [📚 模型库](https://modelscope.cn/models) | [📖 文档](https://modelscope.cn/docs)"
)
