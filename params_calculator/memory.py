from typing import Dict, Tuple


def estimate_vram(param_count: int) -> Dict[str, str]:
    def bytes_to_gb(b):
        return f"{b / (1024 ** 3):.2f} GB"

    return {
        "FP16/BF16 (2 bytes)": bytes_to_gb(param_count * 2),
        "FP8/Int8 (1 byte)": bytes_to_gb(param_count * 1),
        "FP4/Int4 (0.5 byte)": bytes_to_gb(param_count * 0.5),
    }


def estimate_kv_cache(
    model_config,
    context_length: int = 2048,
    batch_size: int = 1,
    dtype: str = "fp16",
    tp: int = 1,
) -> Tuple[str, dict]:
    num_layers = getattr(model_config, "num_hidden_layers", -1)
    num_attention_heads = getattr(model_config, "num_attention_heads", -1)
    num_key_value_heads = getattr(model_config, "num_key_value_heads", -1)
    hidden_size = getattr(model_config, "hidden_size", -1)
    head_dim = getattr(model_config, "head_dim", -1)

    if num_attention_heads < 0 or num_key_value_heads < 0 or hidden_size < 0:
        return "0.00 GB", {}
    if head_dim < 0:
        head_dim = hidden_size // num_attention_heads

    dtype_size = {
        "fp32": 4,
        "fp16": 2,
        "bf16": 2,
        "fp8": 1,
        "int8": 1,
        "int4": 0.5,
    }.get(dtype.lower(), 2)

    calculation_steps = {
        "层数": num_layers,
        "KV注意力头数": num_key_value_heads,
        "每个头的维度": head_dim,
        "上下文长度": context_length,
        "批大小": batch_size,
        "数据类型字节数": dtype_size,
        "KV向量数": 2,
        "张量并行度": tp,
    }

    kv_cache_bytes = (
        num_layers
        * num_key_value_heads
        * head_dim
        * context_length
        * 2
        * batch_size
        * dtype_size
    )
    kv_cache_bytes /= tp

    return f"{kv_cache_bytes / (1024 ** 3):.2f} GB", calculation_steps
