import time
import plotly.express as px
import streamlit as st
import pandas as pd
from params_calculator.common import format_number, get_dtype_size
from params_calculator.memory import estimate_vram, estimate_kv_cache
from params_calculator.analysis import analyze_model_structure

# --- 页面配置 ---
st.set_page_config(
    page_title="ModelScope Model Params Calculator", page_icon="🧮", layout="wide"
)


# --- 核心逻辑函数 ---
@st.cache_data(show_spinner=False)
def cached_analyze_model_structure(model_id: str, trust_remote_code: bool):
    return analyze_model_structure(model_id, trust_remote_code)


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
        "Trust Remote Code",
        value=True,
        help="大多数ModelScope模型需要此选项，否则可能无法加载配置",
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
        "上下文长度",
        options=list(context_options.keys()),
        index=4,
        help="选择预设长度或自定义",  # 默认选择 32K
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
    batch_size = st.number_input(
        "批大小", value=8, min_value=1, step=1, help="推理时的批量大小"
    )
    tp = st.number_input(
        "张量并行度 (TP)",
        value=2,
        min_value=1,
        step=1,
        help="模型并行度，通常用于多卡推理",
    )
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

    success, info, df, error_msg, config = analyze_model_structure(
        model_input, trust_remote
    )

    if success:
        elapsed_time = time.time() - start_time
        status_container.update(
            label=f"✅ 分析完成！耗时 {elapsed_time:.2f}秒",
            state="complete",
            expanded=False,
        )

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

            val = info.get("validation")
            if val and val.get("formula_total", 0) and val.get("actual_total", 0):
                mcols = st.columns(3)
                mcols[0].metric("公式总计", format_number(val["formula_total"]))
                mcols[1].metric("实际枚举总计", format_number(val["actual_total"]))
                mcols[2].metric(
                    "差异", format_number(abs(val["delta"])) if val["delta"] else "0"
                )
                if not val.get("match", False):
                    st.warning(
                        "公式与实际枚举存在>1%的差异，请检查架构特殊项（Bias/共享权重/特殊归一化等）。"
                    )

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
                    col12.metric(
                        "激活参数",
                        format_number(detail_info["详细计算"]["总计（激活参数）"]),
                    )

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
                kv_size, kv_steps = estimate_kv_cache(
                    config, context_length, batch_size, dtype_select, tp
                )
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
                        st.latex(
                            r"""\text{KB} = \frac{L \times H \times D \times C \times 2 \times B \times S}{TP}"""
                        )

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

            # 基础信息表格
            if "基础信息" in detail_info:
                st.write("### 模型配置信息")
                base_info = detail_info["基础信息"]
                base_rows = [
                    {"项目": str(k), "值": str(v)} for k, v in base_info.items()
                ]
                df_base = pd.DataFrame(base_rows)
                df_base["项目"] = df_base["项目"].astype(str)
                df_base["值"] = df_base["值"].astype(str)
                st.dataframe(df_base, width=1000)

            # 结构化公式展示
            if "公式" in detail_info:
                st.write("### 分段公式")

                def parse_formula(lines):
                    blocks = []
                    current = {"title": None, "sections": [], "lines": []}
                    sub = None

                    def flush_sub():
                        nonlocal sub, current
                        if sub and sub.get("lines"):
                            current["sections"].append(
                                {"subtitle": sub["subtitle"], "lines": sub["lines"]}
                            )
                            sub = None

                    def flush_block():
                        nonlocal current, blocks
                        flush_sub()
                        if current["title"] or current["lines"] or current["sections"]:
                            blocks.append(current)
                        current = {"title": None, "sections": [], "lines": []}

                    for raw in lines:
                        line = raw.strip()
                        if not line:
                            continue
                        if line.startswith("### "):
                            flush_block()
                            current["title"] = line[4:]
                        elif line.startswith("#### "):
                            flush_sub()
                            sub = {"subtitle": line[5:], "lines": []}
                        else:
                            if sub is not None:
                                sub["lines"].append(line)
                            else:
                                current["lines"].append(line)
                    flush_block()
                    return blocks

                blocks = parse_formula(detail_info["公式"])
                for blk in blocks:
                    if blk["title"]:
                        st.markdown(f"**{blk['title']}**")
                    if blk["lines"]:
                        st.code("\n".join(blk["lines"]), language="text")
                    for sec in blk["sections"]:
                        st.markdown(f"*{sec['subtitle']}*")
                        if sec["lines"]:
                            st.code("\n".join(sec["lines"]), language="text")

            # 关键分项表格
            calc = detail_info.get("详细计算", {})
            if calc:
                st.write("### 关键分项")
                rows = []
                keys = [
                    "Embedding参数",
                    "每层Attention参数",
                    "每层MoE总参数",
                    "每层MLP参数",
                    "每层归一化参数",
                    "每层总参数(MoE)",
                    "每层总参数(Dense)",
                    "所有层总参数",
                    "所有层激活参数",
                    "LM Head参数",
                    "总计",
                    "总计（激活参数）",
                ]
                for k in keys:
                    if k in calc:
                        val = calc[k]
                        try:
                            if isinstance(val, (int, float)):
                                sval = format_number(int(val))
                            else:
                                sval = str(val)
                        except Exception:
                            sval = str(val)
                        rows.append({"项目": k, "值": sval})
                if rows:
                    df_rows = pd.DataFrame(rows)
                    df_rows["项目"] = df_rows["项目"].astype(str)
                    df_rows["值"] = df_rows["值"].astype(str)
                    st.dataframe(df_rows, width=1000)

            # LaTeX 总结（MoE/Dense）
            st.write("### 公式总结")
            st.latex(
                r"""
                \begin{aligned}
                \text{Dense: } & \text{总} = V\cdot H + N \cdot (4H^2 + 3HI + \text{Norm}) + \text{LM} \\
                \text{MoE: } & \text{总} = V\cdot H + N_{moe} \cdot (4H^2 + (H\cdot E_{gate} + 3H\sum_i E_i N_i) + \text{Norm}) + N_{dense} \cdot (4H^2 + 3HI + \text{Norm}) + \text{LM}
                \end{aligned}
                """
            )

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
                    type_stats["Percentage"] = (
                        type_stats["Count"] / type_stats["Count"].sum() * 100
                    ).round(2)
                    st.dataframe(
                        type_stats[["ParamType", "Count", "Percentage"]],
                        width="stretch",
                    )

                # 层参数统计
                st.write("### 每层参数统计")
                layer_stats = (
                    df[df["LayerIdx"] >= 0]
                    .groupby("LayerIdx")["Count"]
                    .sum()
                    .reset_index()
                )
                layer_stats = layer_stats.sort_values("LayerIdx")

                if not layer_stats.empty:
                    # 计算平均值
                    avg_params_per_layer = layer_stats["Count"].mean()
                    st.info(f"平均每层参数: {format_number(int(avg_params_per_layer))}")

                    # 显示层参数表格
                    st.dataframe(layer_stats, width="stretch")

                # 详细参数表
                st.write("### 完整参数列表")
                st.dataframe(
                    df[["Full Name", "Shape", "Count", "ParamType", "LayerIdx"]],
                    width="stretch",
                    height=500,
                )
            else:
                st.warning("未能解析出详细参数结构。")

        with tab_viz:
            st.subheader("📈 参数可视化")

            if not df.empty:
                # 1. 参数类型分布饼图
                col1, col2 = st.columns(2)

                with col1:
                    type_df = df.groupby("ParamType")["Count"].sum().reset_index()
                    fig1 = px.pie(
                        type_df,
                        values="Count",
                        names="ParamType",
                        title="参数类型分布",
                        hole=0.3,
                    )
                    st.plotly_chart(fig1, width="stretch")

                with col2:
                    # 2. 层级分布条形图
                    if df["LayerIdx"].max() > 0:
                        layer_df = (
                            df[df["LayerIdx"] >= 0]
                            .groupby("LayerIdx")["Count"]
                            .sum()
                            .reset_index()
                        )
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
                df_grouped = (
                    df.groupby(["Group", "LayerIdx", "SubGroup", "ParamType"])["Count"]
                    .sum()
                    .reset_index()
                )
                df_grouped = df_grouped.sort_values(["Group", "LayerIdx"])

                fig3 = px.treemap(
                    df_grouped,
                    path=[
                        px.Constant(model_input),
                        "Group",
                        "LayerIdx",
                        "ParamType",
                        "SubGroup",
                    ],
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
            st.markdown(
                "🔍 你可以在 [ModelScope](https://modelscope.cn/models) 搜索模型"
            )

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
