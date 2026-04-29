import streamlit as st
import sys
import os
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
import warnings

warnings.filterwarnings("ignore")

# ====================== 项目路径 ======================
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from datasets import linear, lotka_volterra, lorenz96, swat, nonlinear, msds
from args import linear_args, lotka_volterra_args, lorenz96_args, swat_args, msds_args, nonlinear_args
from models import aerca

st.set_page_config(page_title="AERCA 可视化演示", page_icon="🔍", layout="wide")

st.title("🚀 AERCA 多变量时间序列根因分析可视化平台")

# ====================== 侧边栏 ======================
st.sidebar.header("🎛️ 控制面板")

dataset_name = st.sidebar.selectbox(
    "选择数据集",
    ["linear", "lotka_volterra", "lorenz96", "msds", "swat", "nonlinear"],
    index=0
)

# ====================== 异常类型选择（仅 linear 数据集有效） ======================
if dataset_name == "linear":
    adtype = st.sidebar.selectbox(
        "异常类型",
        ["spike", "step", "causal"],
        format_func=lambda x: {
            "spike": "Spike（突然尖峰异常）",
            "step": "Step（阶跃持续异常）",
            "causal": "Causal Propagation（因果传播异常）"
        }[x]
    )
else:
    adtype = None

preprocessing_data = st.sidebar.radio(
    "数据处理方式",
    [1, 0],
    format_func=lambda x: "生成新数据并保存" if x == 1 else "加载已有数据",
    horizontal=True
)

training_aerca = st.sidebar.checkbox("执行模型训练", value=True)
epochs = st.sidebar.slider("训练轮数", 10, 5000, 50, step=10)
lr = st.sidebar.number_input("学习率", value=0.001, format="%.5f")

run_button = st.sidebar.button("🚀 开始运行完整流程（训练+测试）", type="primary", width='stretch')

# ====================== 数据集映射 ======================
dataset_mapping = {
    "linear": {"class": linear.Linear, "args": linear_args.create_arg_parser, "use_slice": True},
    "lotka_volterra": {"class": lotka_volterra.LotkaVolterra, "args": lotka_volterra_args.create_arg_parser, "use_slice": True},
    "lorenz96": {"class": lorenz96.Lorenz96, "args": lorenz96_args.create_arg_parser, "use_slice": True},
    "msds": {"class": msds.MSDS, "args": msds_args.create_arg_parser, "use_slice": False},
    "swat": {"class": swat.SWaT, "args": swat_args.create_arg_parser, "use_slice": False},
    "nonlinear": {"class": nonlinear.Nonlinear, "args": nonlinear_args.create_arg_parser, "use_slice": True},
}

# ====================== Session State ======================
if "data_class" not in st.session_state:
    st.session_state.data_class = None
if "dataset_name" not in st.session_state:
    st.session_state.dataset_name = None
if "options" not in st.session_state:
    st.session_state.options = None
if "has_run_model" not in st.session_state:
    st.session_state.has_run_model = False
if "test_x_ab" not in st.session_state:
    st.session_state.test_x_ab = None
if "test_label" not in st.session_state:
    st.session_state.test_label = None
if "test_causal" not in st.session_state:
    st.session_state.test_causal = None
if "root_cause_results" not in st.session_state:
    st.session_state.root_cause_results = None
if "causal_results" not in st.session_state:
    st.session_state.causal_results = None

# ====================== 主逻辑 ======================
if run_button:
    with st.spinner(f"正在处理 {dataset_name} 数据集..."):
        try:
            parser = dataset_mapping[dataset_name]["args"]()
            args = parser.parse_args([])
            options = vars(args)
            options.update({
                'dataset_name': dataset_name,
                'preprocessing_data': preprocessing_data,
                'training_aerca': training_aerca,
                'epochs': epochs,
                'lr': lr,
                'seed': 42,
                'adtype': adtype if dataset_name == "linear" else "non_causal",
            })

            DataClass = dataset_mapping[dataset_name]["class"]
            data_class = DataClass(options)

            if preprocessing_data == 1:
                data_class.generate_example()
                data_class.save_data()

                if dataset_name == "linear":
                    st.success(f"✅ 数据生成完成！异常类型：{adtype}")
                else:
                    st.success(f"✅ 数据生成完成！异常类型：non_causal")
            else:
                data_class.load_data()
                st.success("✅ 数据加载完成！")

            st.session_state.data_class = data_class
            st.session_state.dataset_name = dataset_name
            st.session_state.options = options

            st.success("✅ 数据准备完成！下方可查看可视化结果。")

        except Exception as e:
            st.error(f"数据处理出错: {str(e)}")

# ====================== 数据可视化部分 ======================
if st.session_state.data_class is not None:
    data_class = st.session_state.data_class
    current_dataset = st.session_state.dataset_name
    options = st.session_state.options

    st.subheader(f"📊 数据集可视化 - {current_dataset.upper()}")

    # ====================== 使用测试集（全部改为测试集可视化） ======================
    if st.session_state.get('has_run_model', False) and st.session_state.get('test_x_ab') is not None:
        test_x_ab = st.session_state.test_x_ab
        test_label = st.session_state.test_label
        test_size = len(test_x_ab)
        num_vars = test_x_ab.shape[2]
    else:
        test_size = options.get('testing_size', 100)
        num_vars = options.get('num_vars', 4)
        test_x_ab = data_class.data_dict.get('x_ab_list')
        test_label = data_class.data_dict.get('label_list')

    st.caption(f"**测试集样本**（共 {test_size} 个样本，包含异常，用于可视化和评估）")

    test_sample_idx = st.slider(
        "选择测试样本索引",
        0,
        test_size - 1,
        0,
        key="test_sample_idx_slider"
    )

    # ==================== 一键可视化异常变化（测试集） ====================
    st.markdown("### 👁️‍🗨️ 一键可视化异常变化（测试集）")

    if st.button("🔍 一键显示正常 vs 异常对比", type="primary", width='stretch'):
        if st.session_state.get('has_run_model', False):
            idx_offset = options.get('training_size', 200)
            x_n = data_class.data_dict['x_n_list'][test_sample_idx + idx_offset]
            x_ab = test_x_ab[test_sample_idx]
        else:
            x_n = data_class.data_dict['x_n_list'][test_sample_idx]
            x_ab = data_class.data_dict['x_ab_list'][test_sample_idx]

        df_compare = pd.DataFrame({'时间步': np.arange(len(x_n))})
        for v in range(num_vars):
            df_compare[f'var_{v}'] = x_n[:, v]
            df_compare[f'var_{v}_ab'] = x_ab[:, v]

        fig_compare = go.Figure()
        color_sequence = px.colors.qualitative.Plotly + px.colors.qualitative.Set3 + px.colors.qualitative.Pastel
        colors = [color_sequence[i % len(color_sequence)] for i in range(num_vars)]

        for v in range(num_vars):
            name = f"X_{v}" if current_dataset == "lorenz96" else f"var_{v}"
            fig_compare.add_trace(go.Scatter(
                x=df_compare['时间步'], y=df_compare[f'var_{v}'],
                mode='lines', name=f'{name} (正常)',
                line=dict(color=colors[v], width=1.8, dash='dot')
            ))
            fig_compare.add_trace(go.Scatter(
                x=df_compare['时间步'], y=df_compare[f'var_{v}_ab'],
                mode='lines', name=f'{name} (异常)',
                line=dict(color=colors[v], width=2.5)
            ))

        fig_compare.update_layout(
            title="正常数据 vs 异常数据叠加对比（实线=异常，虚线=正常）",
            height=550,
            xaxis_title="时间步",
            yaxis_title="变量值",
            template="plotly_white",
            legend_title="变量"
        )
        st.plotly_chart(fig_compare, width='stretch')

        st.markdown("**异常差值图**")
        diff = x_ab - x_n
        df_diff = pd.DataFrame(diff, columns=[f'X_{i}' if current_dataset == "lorenz96" else f'var_{i}' for i in range(num_vars)])
        df_diff = df_diff.reset_index().melt(id_vars='index', var_name='变量', value_name='差值')
        df_diff = df_diff.rename(columns={'index': '时间步'})
        fig_diff = px.line(df_diff, x='时间步', y='差值', color='变量',
                           title="异常差值曲线（差值越大表示异常越明显）")
        fig_diff.update_layout(height=400)
        st.plotly_chart(fig_diff, width='stretch')

    # ==================== 正常 / 异常时间序列（全部使用测试集） ====================
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**正常时间序列（测试集）**")
        if st.session_state.get('has_run_model', False):
            idx_offset = options.get('training_size', 200)
            x_n = data_class.data_dict['x_n_list'][test_sample_idx + idx_offset]
        else:
            x_n = data_class.data_dict['x_n_list'][test_sample_idx]

        if current_dataset == "lotka_volterra":
            p = num_vars // 2
            var_names = [f"Prey_{i}" for i in range(p)] + [f"Predator_{i}" for i in range(p)]
        elif current_dataset == "lorenz96":
            var_names = [f"X_{i}" for i in range(num_vars)]
        else:
            var_names = [f'var_{i}' for i in range(num_vars)]

        df_n = pd.DataFrame(x_n, columns=var_names)
        df_n = df_n.reset_index().melt(id_vars='index', var_name='变量', value_name='值')
        df_n = df_n.rename(columns={'index': '时间步'})
        fig_n = px.line(df_n, x='时间步', y='值', color='变量', title="正常数据（测试集）")
        st.plotly_chart(fig_n, width='stretch')

    with col2:
        st.markdown("**异常时间序列（测试集）**")
        if st.session_state.get('has_run_model', False) and st.session_state.get('test_x_ab') is not None:
            x_ab = st.session_state.test_x_ab[test_sample_idx]
        else:
            x_ab = data_class.data_dict['x_ab_list'][test_sample_idx]

        if current_dataset == "lotka_volterra":
            p = num_vars // 2
            var_names = [f"Prey_{i}" for i in range(p)] + [f"Predator_{i}" for i in range(p)]
        elif current_dataset == "lorenz96":
            var_names = [f"X_{i}" for i in range(num_vars)]
        else:
            var_names = [f'var_{i}' for i in range(num_vars)]

        df_ab = pd.DataFrame(x_ab, columns=var_names)
        df_ab = df_ab.reset_index().melt(id_vars='index', var_name='变量', value_name='值')
        df_ab = df_ab.rename(columns={'index': '时间步'})
        fig_ab = px.line(df_ab, x='时间步', y='值', color='变量', title="异常数据（测试集）")
        st.plotly_chart(fig_ab, width='stretch')

    # 异常标签热力图（测试集）
    st.markdown("**异常位置热力图**")
    if st.session_state.get('has_run_model', False) and st.session_state.get('test_label') is not None:
        label_sample = st.session_state.test_label[test_sample_idx]
    else:
        label_sample = data_class.data_dict['label_list'][test_sample_idx]

    fig_label = px.imshow(label_sample.T,
                          labels=dict(x="时间步", y="变量", color="是否异常"),
                          title="异常位置热力图 (1 = 异常)",
                          color_continuous_scale="Blues", aspect="auto")
    fig_label.update_layout(height=500, margin=dict(l=120, r=50, t=80, b=60))
    st.plotly_chart(fig_label, width='stretch')

    # ====================== 变量因果关系图 ======================
    st.markdown("**变量因果关系图**")
    if 'causal_struct' in data_class.data_dict:
        causal_matrix = data_class.data_dict['causal_struct']

        if current_dataset == "lotka_volterra":
            p = num_vars // 2
            fig_causal = px.imshow(causal_matrix,
                                   title="Lotka-Volterra 因果矩阵",
                                   color_continuous_scale="Blues",
                                   labels=dict(x="被影响变量", y="影响变量"))
            st.markdown(f"**生物解释**：前 {p} 个变量为 **猎物 (Prey)**，后 {p} 个变量为 **捕食者 (Predator)**。存在明显的捕食者-猎物因果交互。")
        elif current_dataset == "lorenz96":
            fig_causal = px.imshow(causal_matrix,
                                   title="Lorenz96 因果矩阵",
                                   color_continuous_scale="Blues",
                                   labels=dict(x="被影响变量", y="影响变量"))
            st.markdown("**混沌系统解释**：40变量环形因果系统，每个变量受到前两个变量的影响，同时影响后两个变量，形成强耦合循环混沌动力学。")
        else:
            fig_causal = px.imshow(causal_matrix,
                                   title="真实因果矩阵",
                                   color_continuous_scale="Blues",
                                   labels=dict(x="被影响变量", y="影响变量"))

        st.plotly_chart(fig_causal, width='stretch')
    else:
        st.info("该数据集暂无因果结构信息")

    # ====================== AERCA 模型运行与结果可视化 ======================
    st.subheader("🔬 AERCA 模型运行与结果分析")

    if st.button("🚀 运行 AERCA 模型训练与根因分析", type="primary", use_container_width=True):
        with st.spinner("正在按照 main.py 逻辑完整运行模型（训练 + 测试）..."):
            try:
                argv = ["main.py", "--dataset_name", current_dataset, "--return_results"]
                if current_dataset == "linear" and adtype:
                    argv.extend(["--adtype", adtype])

                original_argv = sys.argv.copy()
                sys.argv = argv

                from main import main as run_main

                results = run_main(sys.argv)

                if results is None:
                    st.error("main.py 未返回结果，请检查 main.py 是否已按方案B修改")
                else:
                    st.success("✅ 模型训练与根因分析完成！")
                    st.info(f"当前使用的异常类型：{adtype if current_dataset == 'linear' else 'non_causal'}")

                    st.session_state.test_x_ab = results['test_x_ab']
                    st.session_state.test_label = results['test_label']
                    st.session_state.test_causal = results.get('test_causal')
                    st.session_state.root_cause_results = results['root_cause_results']
                    st.session_state.causal_results = results['causal_results']
                    st.session_state.has_run_model = True
                    st.session_state.current_dataset = current_dataset

                    st.success("✅ 所有结果已准备好，可在下方查看可视化")

            except Exception as e:
                st.error(f"运行出错: {str(e)}")
                st.exception(e)
            finally:
                sys.argv = original_argv

    # ====================== 高亮可视化 + 因果发现 ======================
    if st.session_state.get('has_run_model', False) and st.session_state.get('test_x_ab') is not None:
        st.subheader("🎯 根因分析效果一览")
        st.markdown("**模型在异常数据中找出“根本原因”的能力评估**")

        root_results = st.session_state.root_cause_results

        col1, col2 = st.columns(2)
        with col1:
            st.metric("✅ 变量定位准确率", f"{root_results['ac_at'][2]:.1%}", "Top-5")
        with col2:
            st.metric("⏱️ 时间+变量联合准确率", f"{root_results['ac_star_at'][1]:.1%}", "Top-10")

        col3, col4, col5, col6 = st.columns(4)
        with col3:
            st.metric("Top-1 准确率", f"{root_results['ac_at'][0]:.1%}")
        with col4:
            st.metric("Top-10 平均排名", f"{root_results['avg_at_10']:.2f}")
        with col5:
            st.metric("严格联合准确率 (Top-1)", f"{root_results['ac_star_at'][0]:.1%}")
        with col6:
            st.metric("严格联合准确率 (Top-100)", f"{root_results['ac_star_at'][2]:.1%}")

        st.markdown("**异常发生时刻与模型预测根因高亮**")

        test_x_ab = st.session_state.test_x_ab
        test_label = st.session_state.test_label
        results = st.session_state.root_cause_results

        num_vars = test_x_ab.shape[2]

        if current_dataset == "lotka_volterra":
            p = num_vars // 2
            var_names = [f"Prey_{i}" for i in range(p)] + [f"Predator_{i}" for i in range(p)]
        elif current_dataset == "lorenz96":
            var_names = [f"X_{i}" for i in range(num_vars)]
        else:
            var_names = [f'var_{i}' for i in range(num_vars)]

        test_sample_idx = st.slider("选择测试样本索引", 0, len(test_x_ab) - 1, 0, key="test_sample_slider")

        x_ab = test_x_ab[test_sample_idx]
        time_steps = np.arange(len(x_ab))

        fig_highlight = go.Figure()

        color_sequence = px.colors.qualitative.Plotly + px.colors.qualitative.Set3 + px.colors.qualitative.Pastel
        colors = [color_sequence[i % len(color_sequence)] for i in range(num_vars)]

        for i in range(num_vars):
            fig_highlight.add_trace(go.Scatter(
                x=time_steps,
                y=x_ab[:, i],
                mode='lines',
                name=var_names[i],
                line=dict(color=colors[i], width=1.5)   # Lorenz96 降低线宽，减少重叠
            ))

        label = test_label[test_sample_idx]
        anomaly_mask = label.sum(axis=1) > 0
        if anomaly_mask.any():
            starts = np.where(np.diff(anomaly_mask.astype(int)) == 1)[0] + 1
            ends = np.where(np.diff(anomaly_mask.astype(int)) == -1)[0] + 1
            if len(starts) == 0 and anomaly_mask[0]:
                starts = [0]
            if len(ends) == 0 and anomaly_mask[-1]:
                ends = [len(anomaly_mask)]

            for s, e in zip(starts, ends):
                fig_highlight.add_vrect(
                    x0=s, x1=e,
                    fillcolor="orange", opacity=0.25,
                    line_width=0,
                    annotation_text="真实异常区间",
                    annotation_position="top left"
                )

        if results.get('predicted_root_causes'):
            pred = results['predicted_root_causes'][test_sample_idx]
            var_name = var_names[pred['root_cause_var_idx']]
            fig_highlight.add_vline(
                x=pred['root_cause_time'],
                line_dash="dash",
                line_color="red",
                line_width=3,
                annotation_text=f"模型预测根因: {var_name}",
                annotation_position="bottom right"
            )

        fig_highlight.update_layout(
            title="异常时间序列 + 根因高亮",
            height=550,
            xaxis_title="时间步",
            yaxis_title="变量值",
            template="plotly_white",
            legend_title="变量"
        )
        st.plotly_chart(fig_highlight, width='stretch')

        # 因果发现结果
        st.subheader("🔗 因果发现结果可视化")
        causal_results = st.session_state.causal_results
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("因果关系 F1 分数", f"{causal_results['f1_mean']:.4f}")
        with col2:
            st.metric("AUROC", f"{causal_results['auroc_mean']:.4f}")
        with col3:
            st.metric("AUPRC", f"{causal_results['auprc_mean']:.4f}")
        with col4:
            st.metric("Hamming Distance", f"{causal_results['hamming_mean']:.4f}")

        st.markdown("**真实因果矩阵 vs 模型预测因果矩阵**")
        col_true, col_pred = st.columns(2)
        with col_true:
            st.markdown("**真实因果结构**")
            fig_true = px.imshow(causal_results['true_causal_matrix'], title="真实因果矩阵",
                                 color_continuous_scale="Blues")
            st.plotly_chart(fig_true, width='stretch')
        with col_pred:
            st.markdown("**模型预测因果结构**")
            fig_pred = px.imshow(causal_results['predicted_causal_matrix'], title="模型预测因果矩阵",
                                 color_continuous_scale="Blues")
            st.plotly_chart(fig_pred, width='stretch')

    else:
        st.info("👈 请先生成数据，然后点击上方按钮运行模型")

st.sidebar.caption("AERCA Streamlit 可视化 Demo")