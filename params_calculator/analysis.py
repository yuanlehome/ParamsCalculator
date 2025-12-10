import os
import re
import json
import sqlite3
import time
import transformers
from accelerate import init_empty_weights
import numpy as np
import torch
from params_calculator.common import ALIASES
from typing import Any, Tuple

import pandas as pd
from modelscope import AutoConfig, AutoModel

from params_calculator.common import format_number


def extract_layer_index(name: str) -> int:
    match = re.search(r"layers.(\d+)", name)
    return int(match.group(1)) if match else -1


def identify_param_type(name: str) -> str:
    n = name.lower()
    if "embedding" in n:
        return "embedding"
    if any(k in n for k in ["q_proj", "k_proj", "v_proj", "attn", "attention"]):
        return "attention"
    if any(k in n for k in ["mlp", "fc", "gate", "up_proj", "down_proj"]):
        return "mlp"
    if ("norm" in n) or ("ln" in n):
        return "norm"
    if ("lm_head" in n) or ("head" in n):
        return "head"
    return "other"


def _bool(config: Any, *keys: str) -> bool:
    for k in keys:
        v = getattr(config, k, None)
        if isinstance(v, bool):
            return v
    return False


def _norm_is_rms(config: Any) -> bool:
    if _bool(config, "use_rms_norm"):
        return True
    if hasattr(config, "rms_norm_eps"):
        return True
    t = str(getattr(config, "norm_type", "")).lower()
    return "rms" in t


def _tie_embeddings(config: Any) -> bool:
    keys = ["tie_word_embeddings", "tie_embeddings", "lm_head_tied"]
    return any(_bool(config, k) for k in keys)


def _get_attention_heads(config: Any):
    h = getattr(config, "num_attention_heads", 0)
    kv = getattr(config, "num_key_value_heads", h)
    return {"num_attention_heads": h, "num_key_value_heads": kv}


def _get_mlp_sizes(config: Any):
    hidden_size = getattr(config, "hidden_size", 0)
    intermediate_size = getattr(config, "intermediate_size", 0)
    moe_intermediate_size = getattr(
        config,
        "moe_intermediate_size",
        getattr(
            config,
            "expert_intermediate_size",
            getattr(config, "ffn_hidden_size", intermediate_size),
        ),
    )
    return {
        "hidden_size": hidden_size,
        "intermediate_size": intermediate_size,
        "moe_intermediate_size": moe_intermediate_size,
    }


# 别名映射解析
def _alias_get(config: Any, keys_or_group, default=None):
    keys = keys_or_group
    if isinstance(keys_or_group, str):
        keys = ALIASES.get(keys_or_group, [])
    for k in keys:
        v = getattr(config, k, None)
        if v is not None:
            return v
    return default


def _get_list_pref(config: Any, keys_or_group):
    keys = keys_or_group
    if isinstance(keys_or_group, str):
        keys = ALIASES.get(keys_or_group, [])
    for k in keys:
        v = getattr(config, k, None)
        if isinstance(v, list) and v:
            return v
    return None


def _resolve_expert_segments(config: Any):
    num_experts_list = _get_list_pref(config, "experts_count")
    shared = int(_alias_get(config, "experts_shared", 0) or 0)
    inter_list = _get_list_pref(config, "expert_intermediate")
    topk = int(_alias_get(config, "experts_per_token", 0) or 0)
    start_idxs = _get_list_pref(config, "moe_layers_start") or []
    end_idxs = _get_list_pref(config, "moe_layers_end") or []
    interval = int(_alias_get(config, "moe_layers_interval", 1) or 1)

    segments = []
    layer_union = set()
    if isinstance(num_experts_list, list) and num_experts_list:
        for i, ne in enumerate(num_experts_list):
            iseg = (
                inter_list[i]
                if isinstance(inter_list, list) and i < len(inter_list)
                else _alias_get(
                    config,
                    ["expert_intermediate_size", "ffn_hidden_size"],
                    getattr(config, "intermediate_size", 0),
                )
            )
            s = start_idxs[i] if i < len(start_idxs) else 0
            e = (
                end_idxs[i]
                if i < len(end_idxs)
                else getattr(config, "num_hidden_layers", 0) - 1
            )
            seg_layers = 0
            if e >= s:
                seg_layers = e - s + 1
                if interval > 1:
                    seg_layers = (seg_layers + interval - 1) // interval
                # 构建分段层索引并合并为并集
                step = max(1, interval)
                for L in range(int(s), int(e) + 1, step):
                    layer_union.add(L)
            segments.append(
                {
                    "num_experts": int(ne),
                    "intermediate": int(iseg),
                    "layers": int(seg_layers),
                    "start": int(s),
                    "end": int(e),
                }
            )
    return {
        "segments": segments,
        "shared_experts": shared,
        "topk": topk,
        "interval": interval,
        "layer_union_count": len(layer_union),
    }


def _attention_params_per_layer(config: Any):
    hs = getattr(config, "hidden_size", 0)
    q = hs * hs
    k = hs * hs
    v = hs * hs
    o = hs * hs
    return {"q": q, "k": k, "v": v, "out": o, "total": q + k + v + o}


def _mlp_params_per_layer_dense(hidden_size: int, intermediate_size: int) -> int:
    return (hidden_size * intermediate_size) * 2 + (intermediate_size * hidden_size)


def _mlp_params_moe_per_layer(
    hidden_size: int, moe_intermediate_size: int, num_experts: int
):
    gate = hidden_size * num_experts
    expert = (hidden_size * moe_intermediate_size) * 2 + (
        moe_intermediate_size * hidden_size
    )
    total = gate + expert * num_experts
    return {"gate": gate, "expert": expert, "total": total}


def calculate_model_params_detail(config: Any):
    details = {}
    formulas = []
    try:
        model_type = str(getattr(config, "model_type", "unknown")).lower()
        vocab_size = getattr(config, "vocab_size", 0)
        hidden_size = getattr(config, "hidden_size", 0)
        num_layers = getattr(config, "num_hidden_layers", 0)
        heads = _get_attention_heads(config)
        num_attention_heads = heads["num_attention_heads"]
        intermediate_size = getattr(config, "intermediate_size", 0)
        num_experts_per_tok = getattr(
            config, "num_experts_per_tok", getattr(config, "top_k", 0)
        )
        mlp = _get_mlp_sizes(config)
        seg_info = _resolve_expert_segments(config)
        segments = seg_info["segments"]
        shared = seg_info["shared_experts"]
        # 计算总专家数：优先分段，其次通用字段，最后路由+共享
        fallback_experts = getattr(
            config, "num_local_experts", getattr(config, "num_experts", 0)
        )
        total_experts = (
            (sum(s["num_experts"] for s in segments) + shared)
            if segments
            else (
                fallback_experts
                if fallback_experts > 0
                else (
                    getattr(config, "n_routed_experts", 0)
                    + getattr(config, "n_shared_experts", 0)
                )
            )
        )
        is_moe_model = total_experts > 0
        # MoE层数：分段则累加，否则按频率与dense替换计算
        if segments:
            # 使用并集层计数，避免分段重叠导致层数重复累加
            moe_layers = seg_info.get(
                "layer_union_count", sum(s["layers"] for s in segments)
            )
        else:
            freq = int(getattr(config, "moe_layer_freq", 1) or 1)
            dense_replace = int(getattr(config, "first_k_dense_replace", 0) or 0)
            if freq <= 1:
                moe_layers = max(0, num_layers - dense_replace)
            else:
                moe_layers = max(0, (num_layers - dense_replace + freq - 1) // freq)
        dense_layers = max(0, num_layers - moe_layers)
        model_type_with_moe = (
            f"{model_type} (MoE)" if is_moe_model else f"{model_type} (Dense)"
        )
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
            details["基础信息"].update(
                {
                    "专家数量": total_experts,
                    "每token专家数": (
                        seg_info["topk"] if seg_info["topk"] else num_experts_per_tok
                    ),
                    "专家中间层维度": mlp["moe_intermediate_size"],
                    "MoE层数": moe_layers,
                    "Dense层数": dense_layers,
                }
            )
        total_params = 0
        embedding_params = vocab_size * hidden_size
        total_params += embedding_params
        formulas.append("### 1. Embedding 层参数")
        formulas.append("词表大小 × 隐藏层维度")
        formulas.append(
            f"{format_number(vocab_size)} × {format_number(hidden_size)} = **{format_number(embedding_params)}**"
        )
        attn = _attention_params_per_layer(config)
        formulas.append("\n### 2. Attention 层参数（每层）")
        formulas.append("#### a) Q 投影")
        formulas.append(
            f"{format_number(hidden_size)} × {format_number(attn['q'] // hidden_size)} = {format_number(attn['q'])}"
        )
        formulas.append("#### b) K 投影（KV多头）")
        formulas.append(
            f"{format_number(hidden_size)} × {format_number(attn['k'] // hidden_size)} = {format_number(attn['k'])}"
        )
        formulas.append("#### c) V 投影（KV多头）")
        formulas.append(
            f"{format_number(hidden_size)} × {format_number(attn['v'] // hidden_size)} = {format_number(attn['v'])}"
        )
        formulas.append("#### d) Output 投影")
        formulas.append(
            f"{format_number(hidden_size)} × {format_number(hidden_size)} = {format_number(attn['out'])}"
        )
        formulas.append("#### e) 每层Attention总参数")
        formulas.append(f"{format_number(attn['total'])}")
        mlp_moe_per_layer = 0
        mlp_dense_per_layer = 0
        active_params_per_layer = 0
        if intermediate_size > 0 or mlp["moe_intermediate_size"] > 0:
            if is_moe_model and moe_layers > 0:
                formulas.append("\n### 3. MoE层参数（每层）")
                # 分段求和或单段回退
                gate_experts = (
                    sum(s["num_experts"] for s in segments) + shared
                    if segments
                    else total_experts
                )
                gate_params = hidden_size * gate_experts
                expert_sum = 0
                per_expert = []
                if segments:
                    for s in segments:
                        seg_calc = _mlp_params_moe_per_layer(
                            hidden_size, s["intermediate"], s["num_experts"]
                        )
                        expert_params = seg_calc["expert"]
                        expert_sum += expert_params * s["num_experts"]
                        per_expert.append(expert_params)
                else:
                    # 单段回退：使用 expert_intermediate_size / ffn_hidden_size / intermediate_size
                    inter = _alias_get(
                        config,
                        "expert_intermediate",
                        getattr(config, "intermediate_size", 0),
                    )
                    single_calc = _mlp_params_moe_per_layer(
                        hidden_size, int(inter), int(total_experts)
                    )
                    expert_params = single_calc["expert"]
                    expert_sum = expert_params * int(total_experts)
                    per_expert = [expert_params]
                mlp_moe_per_layer = gate_params + expert_sum
                formulas.append(f"Gate: {format_number(gate_params)}")
                if per_expert:
                    formulas.append(
                        f"每个专家参数(示例): {format_number(per_expert[0])}"
                    )
                formulas.append(f"所有专家总参数: {format_number(expert_sum)}")
                topk = (
                    seg_info["topk"]
                    if seg_info["topk"]
                    else int(_alias_get(config, "experts_per_token", 0) or 0)
                )
                active_params_per_layer = gate_params + (
                    sum(per_expert) * topk if per_expert else 0
                )
                formulas.append("#### 激活参数（每token）")
                formulas.append(
                    f"Gate + 专家×{topk} = {format_number(active_params_per_layer)}"
                )
                details["MoE信息"] = {
                    "每层总MoE参数": mlp_moe_per_layer,
                    "每个专家参数": per_expert[0] if per_expert else 0,
                    "每token激活参数": active_params_per_layer,
                    "稀疏性": f"{(topk / max(1, gate_experts) * 100):.1f}%",
                }
            if dense_layers > 0:
                mlp_dense_per_layer = _mlp_params_per_layer_dense(
                    hidden_size, intermediate_size
                )
                formulas.append("\n### 3. MLP层参数（每层）- Dense")
                formulas.append(f"{format_number(mlp_dense_per_layer)}")
        if _norm_is_rms(config):
            norm_params_per_layer = hidden_size * 2
            norm_desc = "RMSNorm × 2"
        else:
            norm_params_per_layer = hidden_size * 4
            norm_desc = "LayerNorm(γ+β) × 2"
        formulas.append("\n### 4. 归一化层参数（每层）")
        formulas.append(f"{norm_desc}: {format_number(norm_params_per_layer)}")
        if is_moe_model:
            params_moe_layer = attn["total"] + mlp_moe_per_layer + norm_params_per_layer
            params_dense_layer = (
                attn["total"] + mlp_dense_per_layer + norm_params_per_layer
            )
            if moe_layers > 0:
                active_params_per_layer = (
                    attn["total"] + active_params_per_layer + norm_params_per_layer
                )
                formulas.append("\n### 5. 每层总参数 (MoE)")
                formulas.append(f"总: {format_number(params_moe_layer)}")
                formulas.append(f"激活: {format_number(active_params_per_layer)}")
                details["MoE信息"]["每层激活参数"] = active_params_per_layer
            if dense_layers > 0:
                formulas.append("\n### 5. 每层总参数 (Dense)")
                formulas.append(f"{format_number(params_dense_layer)}")
            all_layers_params = (
                params_moe_layer * moe_layers + params_dense_layer * dense_layers
            )
        else:
            params_per_layer = (
                attn["total"] + mlp_dense_per_layer + norm_params_per_layer
            )
            formulas.append("\n### 5. 每层总参数")
            formulas.append(f"{format_number(params_per_layer)}")
            all_layers_params = params_per_layer * num_layers
        total_params += all_layers_params
        if is_moe_model:
            all_active_params = (
                (details["MoE信息"]["每层激活参数"] * moe_layers)
                if moe_layers > 0
                else 0
            )
            formulas.append(f"\n### 6. 所有{num_layers}层总参数 (MoE/Dense)")
            formulas.append(f"总: {format_number(all_layers_params)}")
            if moe_layers > 0:
                formulas.append(f"激活: {format_number(all_active_params)}")
                details["MoE信息"]["总激活参数"] = all_active_params
        else:
            formulas.append(f"\n### 6. 所有{num_layers}层总参数")
            formulas.append(f"{format_number(all_layers_params)}")
        if _tie_embeddings(config):
            lm_head_params = 0
            formulas.append("\n### 7. 输出层 (LM Head) 参数")
            formulas.append("权重共享：不额外引入参数")
        else:
            lm_head_params = hidden_size * vocab_size
            formulas.append("\n### 7. 输出层 (LM Head) 参数")
            formulas.append(
                f"{format_number(hidden_size)} × {format_number(vocab_size)} = **{format_number(lm_head_params)}**"
            )
        total_params += lm_head_params
        if is_moe_model:
            formulas.append("\n### 8. 模型总参数量 (MoE)")
            formulas.append(
                f"Embedding + 所有层(总) + LM Head = **{format_number(total_params)}**"
            )
            total_active_params = (
                embedding_params
                + (details["MoE信息"].get("总激活参数", 0))
                + lm_head_params
            )
            sparsity = (
                (1 - (total_active_params / total_params)) * 100 if total_params else 0
            )
            formulas.append("#### 激活参数总量")
            formulas.append(f"**{format_number(total_active_params)}**")
            formulas.append("#### 稀疏率")
            formulas.append(f"**{sparsity:.1f}%**")
            details["详细计算"] = {
                "Embedding参数": embedding_params,
                "每层Attention参数": attn["total"],
                "每层MoE总参数": mlp_moe_per_layer,
                "每个专家参数": details["MoE信息"]["每个专家参数"],
                "每层激活MoE参数": details["MoE信息"].get("每层激活参数", 0),
                "每层归一化参数": norm_params_per_layer,
                "每层总参数(MoE)": params_moe_layer if moe_layers > 0 else 0,
                "每层总参数(Dense)": params_dense_layer if dense_layers > 0 else 0,
                "所有层总参数": all_layers_params,
                "所有层激活参数": details["MoE信息"].get("总激活参数", 0),
                "LM Head参数": lm_head_params,
                "总计": total_params,
                "总计（激活参数）": total_active_params,
                "稀疏率": f"{sparsity:.1f}%",
            }
        else:
            formulas.append("\n### 8. 模型总参数量")
            formulas.append(
                f"Embedding + 所有层 + LM Head = **{format_number(total_params)}**"
            )
            details["详细计算"] = {
                "Embedding参数": embedding_params,
                "每层Attention参数": attn["total"],
                "每层MLP参数": mlp_dense_per_layer,
                "每层归一化参数": norm_params_per_layer,
                "每层总参数": params_per_layer,
                "所有层总参数": all_layers_params,
                "LM Head参数": lm_head_params,
                "总计": total_params,
            }
        details["公式"] = formulas
    except Exception as e:
        formulas.append(f"计算详细公式时出错: {str(e)}")
        details["公式"] = formulas
    return details


_DB_PATH = os.path.join("datasets", "config_cache.db")


def _init_db():
    os.makedirs(os.path.dirname(_DB_PATH), exist_ok=True)
    con = sqlite3.connect(_DB_PATH)
    cur = con.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS configs (
            model_id TEXT PRIMARY KEY,
            model_type TEXT,
            config_json TEXT NOT NULL,
            updated_at INTEGER
        )
        """
    )
    con.commit()
    con.close()


def _load_config_dict(model_id: str) -> dict | None:
    _init_db()
    con = sqlite3.connect(_DB_PATH)
    cur = con.cursor()
    cur.execute("SELECT config_json FROM configs WHERE model_id=?", (model_id,))
    row = cur.fetchone()
    con.close()
    if not row:
        return None
    try:
        return json.loads(row[0])
    except Exception:
        return None


def _save_config_dict(model_id: str, cfg_dict: dict):
    _init_db()
    con = sqlite3.connect(_DB_PATH)
    cur = con.cursor()

    def _jsonable(obj):
        if obj is None or isinstance(obj, (bool, int, float, str)):
            return obj
        if isinstance(obj, (list, tuple, set)):
            return [_jsonable(x) for x in obj]
        if isinstance(obj, dict):
            return {str(k): _jsonable(v) for k, v in obj.items()}
        if isinstance(obj, (np.dtype, np.generic)):
            return str(obj)
        if isinstance(obj, torch.dtype):
            s = str(obj)
            return s.replace("torch.", "")
        if hasattr(obj, "value"):
            try:
                return _jsonable(obj.value)
            except Exception:
                return str(obj)
        return str(obj)

    safe_cfg = _jsonable(cfg_dict)
    cur.execute(
        "REPLACE INTO configs(model_id, model_type, config_json, updated_at) VALUES(?, ?, ?, ?)",
        (
            model_id,
            str((cfg_dict.get("model_type", "") or "")),
            json.dumps(safe_cfg, ensure_ascii=False),
            int(time.time()),
        ),
    )
    con.commit()
    con.close()


def _config_from_dict(cfg_dict: dict):
    model_type = str(cfg_dict.get("model_type", "") or "").lower()
    if not model_type:
        raise ValueError("config dict missing model_type")
    mapping = transformers.CONFIG_MAPPING
    if model_type not in mapping:
        # fallback: find by architectures if available
        archs = cfg_dict.get("architectures", []) or []
        if archs:
            # try prefix heuristics
            prefix = str(model_type).split("_")[0]
            for key in mapping.keys():
                if key.startswith(prefix):
                    model_type = key
                    break
        else:
            raise ValueError(f"unknown model_type '{model_type}' for reconstruction")
    cfg_cls = mapping[model_type]
    return cfg_cls.from_dict(cfg_dict)


def _get_or_fetch_config(model_id: str, trust_remote_code: bool):
    cfg_dict = _load_config_dict(model_id)
    if cfg_dict:
        try:
            return _config_from_dict(cfg_dict)
        except Exception:
            pass
    cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=trust_remote_code)
    # huggingface PretrainedConfig has to_dict
    if hasattr(cfg, "to_dict"):
        cfg_dict = cfg.to_dict()
    else:
        cfg_dict = {k: v for k, v in cfg.__dict__.items() if not k.startswith("_")}
    _save_config_dict(model_id, cfg_dict)
    return cfg


def analyze_model_structure(
    model_id: str, trust_remote_code: bool
) -> Tuple[bool, Any, pd.DataFrame, str, Any]:
    try:
        try:
            config = _get_or_fetch_config(model_id, trust_remote_code)
        except Exception as e:
            return False, None, None, f"加载配置失败: {str(e)}", None

        detail_calculation = calculate_model_params_detail(config)

        model = None
        try:
            with init_empty_weights():
                model = AutoModel.from_config(
                    config, trust_remote_code=trust_remote_code
                )
        except Exception:
            model = None
        if model is None:
            try:
                config = AutoConfig.from_pretrained(
                    model_id, trust_remote_code=trust_remote_code
                )
                with init_empty_weights():
                    model = AutoModel.from_config(
                        config, trust_remote_code=trust_remote_code
                    )
            except Exception:
                model = None

        total_params = 0
        trainable_params = 0
        param_data = []
        if model is not None:
            for name, param in model.named_parameters():
                num_params = param.numel()
                total_params += num_params
                if param.requires_grad:
                    trainable_params += num_params
                param_data.append(
                    {
                        "Full Name": name,
                        "Group": (
                            name.split(".")[0] if len(name.split(".")) > 0 else "base"
                        ),
                        "SubGroup": (
                            name.split(".")[1] if len(name.split(".")) > 1 else "other"
                        ),
                        "Shape": str(tuple(param.shape)),
                        "Count": num_params,
                        "Dtype": str(param.dtype).replace("torch.", ""),
                        "LayerIdx": extract_layer_index(name),
                        "ParamType": identify_param_type(name),
                    }
                )
        df_params = pd.DataFrame(param_data)

        calculated_total = detail_calculation.get("详细计算", {}).get("总计", 0)
        final_total = total_params if (total_params > 0) else calculated_total

        info = {
            "model_type": getattr(config, "model_type", "unknown"),
            "total_params": final_total,
            "trainable_params": trainable_params,
            "architectures": getattr(config, "architectures", ["Unknown"]),
            "vocab_size": getattr(config, "vocab_size", "N/A"),
            "hidden_size": getattr(config, "hidden_size", "N/A"),
            "num_layers": getattr(config, "num_hidden_layers", 0),
            "num_heads": getattr(config, "num_attention_heads", 0),
            "max_position_embeddings": getattr(
                config, "max_position_embeddings", "N/A"
            ),
            "intermediate_size": getattr(config, "intermediate_size", "N/A"),
            "source": "ModelScope",
            "detail_calculation": detail_calculation,
            "validation": {
                "formula_total": int(calculated_total),
                "actual_total": int(
                    final_total if model is not None else calculated_total
                ),
                "delta": int(
                    (final_total if model is not None else calculated_total)
                    - calculated_total
                ),
                "match": (
                    True
                    if model is None
                    else (
                        (abs(final_total - calculated_total) / max(1, final_total))
                        < 0.01
                        if calculated_total
                        else False
                    )
                ),
                "enumeration_available": model is not None,
            },
        }

        return True, info, df_params, "", config
    except Exception as e:
        return False, None, None, str(e), None
