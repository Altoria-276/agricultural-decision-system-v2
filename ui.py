import gradio as gr
import tkinter as tk
from tkinter import filedialog
import os
import pandas as pd
from pathlib import Path
from configs.config import Config
from run import run
import ast

SCRIPT_DIR: Path = Path(__file__).parent


def select_excel_file():
    """选择Excel文件"""
    root = tk.Tk()
    root.withdraw()  # 隐藏主窗口
    root.attributes("-topmost", True)  # 确保对话框在最前面

    # 使用项目目录作为起始目录
    project_dir = SCRIPT_DIR / "data"
    file_path = filedialog.askopenfilename(title="选择Excel文件", initialdir=project_dir, filetypes=[("Excel files", "*.xlsx *.xls"), ("All files", "*.*")])

    if file_path:
        return file_path
    return ""


def load_excel_columns(file_path):
    """加载Excel文件的列名"""
    try:
        if not file_path or not Path(file_path).exists():
            return [], [], [], []

        df = pd.read_excel(file_path)
        columns = df.columns.tolist()

        # 返回所有列名供用户选择（4个复选框组）
        return [columns, columns, columns, columns]
    except Exception as e:
        print(f"读取Excel文件错误: {e}")
        return [[], [], [], []]


def parse_vector_input(vector_str):
    """解析用户输入的向量字符串"""
    try:
        # 尝试解析为Python列表格式
        if vector_str.strip():
            vector = ast.literal_eval(vector_str)
            if isinstance(vector, list) and all(isinstance(x, (int, float)) for x in vector):
                return vector
        return None
    except:
        return None


def run_system(excel_file_path, sence_columns, process_columns, type_columns, result_columns, init_vector_str, filter_threshold, search_threshold, search_num_iter, search_multiplier, search_candidates_per_round, search_target_number, select_threshold, select_max_num):
    """运行农业决策系统"""
    try:
        # 验证输入
        if not excel_file_path or not Path(excel_file_path).exists():
            return "请选择有效的Excel文件", None

        # 解析初始向量
        init_vector = parse_vector_input(init_vector_str)
        if init_vector is None:
            return "请输入有效的初始向量（格式：[1, 2, 3, ...]）", None

        # 解析列选择 - Dropdown 多选返回列表，单选返回单个值或 None
        sence_cols = sence_columns if isinstance(sence_columns, list) else []
        process_cols = process_columns if isinstance(process_columns, list) else []
        type_cols = [type_columns] if type_columns else []
        result_cols = [result_columns] if result_columns else []

        if not all([sence_cols, process_cols, type_cols, result_cols]):
            return "请选择所有必需的列（S、P、T、R）", None

        # 创建配置
        config = Config()

        # 更新配置
        config.update("datasets.path", excel_file_path)
        config.update("datasets.sence_columns", sence_cols)
        config.update("datasets.process_columns", process_cols)
        config.update("datasets.type_columns", type_cols)
        config.update("datasets.result_columns", result_cols)
        config.update("init_data", init_vector)
        config.update("filter_threshold", float(filter_threshold))

        # 更新搜索参数
        config.update("search_parameters.threshold", float(search_threshold))
        config.update("search_parameters.num_iter", int(search_num_iter))
        config.update("search_parameters.multiplier", float(search_multiplier))
        config.update("search_parameters.num_candidates_per_round", int(search_candidates_per_round))
        config.update("search_parameters.target_number", float(search_target_number))
        config.update("select_variables_threshold", float(select_threshold))
        config.update("select_variables_max_num", int(select_max_num))

        # 运行系统
        result = run(config)

        if result["success"]:
            message = f"系统运行成功！\n"
            message += f"处理的数据类型: {result['types_processed']}\n"
            message += f"数据形状: {result['data_summary']}"

            # 返回图像路径
            image_path = result["rmse_image"]
            return message, image_path
        else:
            return f"系统运行失败: {result['message']}", None

    except Exception as e:
        return f"运行错误: {str(e)}", None


def ui():
    """
    农业决策系统主界面
    """
    with gr.Blocks(title="农业决策系统 AID v2") as demo:
        gr.Markdown("# 农业决策系统 AID v2")

        # 初始化默认列配置
        default_config = Config()
        # 获取配置文件中的默认列名
        default_sence_columns = default_config["datasets.sence_columns"]
        default_process_columns = default_config["datasets.process_columns"]
        default_type_columns = default_config["datasets.type_columns"]
        default_result_columns = default_config["datasets.result_columns"]
        # 合并所有可能的列名作为choices
        default_columns = list(set(default_sence_columns + default_process_columns + default_type_columns + default_result_columns))

        with gr.Tabs() as tabs:
            # TAB 1: 文件选择和数据配置
            with gr.Tab("数据配置", id="tab1"):
                gr.Markdown("## 步骤1: 选择数据文件和配置列")

                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### 选择Excel文件")
                        excel_file_button = gr.Button("选择Excel文件")
                        excel_file_output = gr.Textbox(label="已选择的Excel文件", value="data/new.xlsx", interactive=False)

                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### 列配置")
                        # 使用动态更新
                        sence_columns = gr.Dropdown(choices=default_columns, value=["降雨量/mm", "坡度/°", "土壤氮含量（g/kg）", "氮肥利用效率%", "NDVI(植被覆盖)", "植被类型", "土地利用类型"], label="场景列 (S - Sence Columns)", info="选择描述环境场景的列（可多选）", multiselect=True)
                        process_columns = gr.Dropdown(choices=default_columns, value=["用量（株/平方米）", "氮肥施肥量（kg/亩)", "磷肥施肥量（kg/亩)"], label="过程列 (P - Process Columns)", info="选择描述处理过程的列（可多选）", multiselect=True)
                        type_columns = gr.Dropdown(choices=default_columns, value="技术分类", label="类型列 (T - Type Columns)", info="选择分类类型的列（通常为单列）")
                        result_columns = gr.Dropdown(choices=default_columns, value="入水口水中氮含量降低程度（%）", label="结果列 (R - Result Columns)", info="选择目标结果列")

                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### 初始参数")
                        init_vector = gr.Textbox(label="初始向量", value="[1501, 2, 14, 26, 0.6136, 2, 3]", info="输入用于相似度计算的初始向量，格式：[val1, val2, ...]")
                        filter_threshold = gr.Number(label="过滤阈值", value=-1, info="余弦相似度过滤阈值")

                tab1_next_button = gr.Button("下一步", variant="primary")

                # 事件绑定
                def update_columns(file_path):
                    columns = load_excel_columns(file_path)
                    # 为每个 Dropdown 组件返回 choices 列表
                    # 分别是 sence_columns, process_columns, type_columns, result_columns 的 choices
                    if columns[0]:  # 如果成功加载了列名
                        return (file_path, gr.update(choices=columns[0]), gr.update(choices=columns[1]), gr.update(choices=columns[2]), gr.update(choices=columns[3]))
                    else:
                        # 如果没有成功加载列名，返回默认列名
                        return (file_path, gr.update(choices=default_columns), gr.update(choices=default_columns), gr.update(choices=default_columns), gr.update(choices=default_columns))

                excel_file_button.click(fn=select_excel_file, outputs=excel_file_output)

                excel_file_output.change(fn=update_columns, inputs=excel_file_output, outputs=[excel_file_output, sence_columns, process_columns, type_columns, result_columns])

                tab1_next_button.click(fn=lambda: gr.update(selected="tab2"), outputs=tabs)

            # TAB 2: 参数设置
            with gr.Tab("参数设置", id="tab2"):
                gr.Markdown("## 步骤2: 设置算法参数")

                with gr.Accordion("搜索参数设置", open=True):
                    with gr.Row():
                        with gr.Column():
                            search_threshold = gr.Number(label="搜索阈值", value=0, info="参数搜索的阈值")
                            search_num_iter = gr.Number(label="迭代次数", value=4, info="搜索迭代次数")
                        with gr.Column():
                            search_multiplier = gr.Number(label="目标值乘数", value=0.95, info="目标数值的乘数因子")
                            search_candidates_per_round = gr.Number(label="每轮候选参数数量", value=5, info="每轮选择的候选参数数量")
                    search_target_number = gr.Number(label="目标数值", value=20, info="搜索目标数值")

                with gr.Accordion("特征选择参数", open=True):
                    with gr.Row():
                        with gr.Column():
                            select_threshold = gr.Number(label="特征选择阈值", value=0.5, info="特征选择的重要性阈值")
                            select_max_num = gr.Number(label="最大特征数量", value=5, info="选择的最大特征数量")

                with gr.Row():
                    tab2_prev_button = gr.Button("上一步")
                    tab2_next_button = gr.Button("下一步", variant="primary")

                tab2_prev_button.click(fn=lambda: gr.update(selected="tab1"), outputs=tabs)

                tab2_next_button.click(fn=lambda: gr.update(selected="tab3"), outputs=tabs)

            # TAB 3: 运行和结果
            with gr.Tab("运行结果", id="tab3"):
                gr.Markdown("## 步骤3: 运行系统并查看结果")

                with gr.Row():
                    with gr.Column():
                        run_button = gr.Button("运行系统", variant="primary", size="lg")

                with gr.Row():
                    with gr.Column():
                        status_output = gr.Textbox(label="运行状态", lines=10, max_lines=15, info="显示系统运行状态和结果")

                with gr.Row():
                    with gr.Column():
                        image_output = gr.Image(label="RMSE对比图")

                with gr.Row():
                    tab3_prev_button = gr.Button("上一步")

                # 运行按钮事件
                run_button.click(
                    fn=run_system,
                    inputs=[excel_file_output, sence_columns, process_columns, type_columns, result_columns, init_vector, filter_threshold, search_threshold, search_num_iter, search_multiplier, search_candidates_per_round, search_target_number, select_threshold, select_max_num],
                    outputs=[status_output, image_output],
                )

                tab3_prev_button.click(fn=lambda: gr.update(selected="tab2"), outputs=tabs)

    return demo


if __name__ == "__main__":
    demo = ui()
    demo.launch()
