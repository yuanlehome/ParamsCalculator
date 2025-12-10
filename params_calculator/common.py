def format_number(num: int) -> str:
    if num >= 1_000_000_000:
        return f"{num / 1_000_000_000:.2f} B"
    elif num >= 1_000_000:
        return f"{num / 1_000_000:.2f} M"
    else:
        return f"{num:,}"


def get_dtype_size(dtype: str) -> float:
    dtype_map = {"fp32": 4, "fp16": 2, "bf16": 2, "fp8": 1, "int8": 1, "int4": 0.5}
    return dtype_map.get(dtype.lower(), 2)


# 字段别名映射（集中管理）
ALIASES = {
    "experts_count": [
        "num_experts",
        "num_local_experts",
        "n_routed_experts",
        "moe_num_experts",  # list
    ],
    "experts_shared": [
        "moe_num_shared_experts",
        "n_shared_experts",
    ],
    "experts_per_token": [
        "moe_k",
        "num_experts_per_tok",
        "top_k",
    ],
    "expert_intermediate": [
        "moe_intermediate_size",  # list
        "expert_intermediate_size",
        "ffn_hidden_size",
        "intermediate_size",
    ],
    "moe_layers_interval": [
        "moe_layer_interval",
        "moe_layer_freq",
    ],
    "moe_layers_start": [
        "moe_layer_start_index",
    ],
    "moe_layers_end": [
        "moe_layer_end_index",
    ],
    "norm_rms": [
        "use_rms_norm",
        "rms_norm_eps",
        "norm_type",
    ],
}
