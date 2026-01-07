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
    # 尝试解析为Python列表格式
    if vector_str.strip():
        vector = ast.literal_eval(vector_str)
        if isinstance(vector, list) and all(isinstance(x, (int, float)) for x in vector):
            return vector
    return None


def run_system(excel_file_path, sence_columns, process_columns, type_columns, result_columns, init_vector_str, filter_threshold, search_threshold, search_num_iter, search_multiplier, search_candidates_per_round, search_target_number, select_threshold, select_max_num):
    """运行农业决策系统"""
    try:
        # 验证输入
        if not excel_file_path or not Path(excel_file_path).exists():
            return "请选择有效的Excel文件", None, "", {}, {}

        # 解析初始向量
        init_vector = parse_vector_input(init_vector_str)
        if init_vector is None:
            return "请输入有效的初始向量（格式：[1, 2, 3, ...]）", None, "", {}, {}

        # 解析列选择 - Dropdown 多选返回列表，单选返回单个值或 None
        sence_cols = sence_columns if isinstance(sence_columns, list) else []
        process_cols = process_columns if isinstance(process_columns, list) else []
        type_cols = [type_columns] if type_columns else []
        result_cols = [result_columns] if result_columns else []

        if not all([sence_cols, process_cols, type_cols, result_cols]):
            return "请选择所有必需的列（S、P、T、R）", None, "", {}, {}

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

            # 返回所有需要的结果
            return (message, result["rmse_image"], result["filter_log"], result["shap_images"], result["best_params_results"])
        else:
            return f"系统运行失败: {result['message']}", None, "", {}, {}

    except Exception as e:
        return f"运行错误: {str(e)}", None, "", {}, {}


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
            with gr.Tab("运行控制", id="tab3"):
                gr.Markdown("## 步骤3: 运行系统")

                with gr.Row():
                    with gr.Column():
                        run_button = gr.Button("运行系统", variant="primary", size="lg")

                with gr.Row():
                    with gr.Column():
                        status_output = gr.Textbox(label="运行状态", lines=8, max_lines=15, info="显示系统运行状态和结果")

                with gr.Row():
                    tab3_prev_button = gr.Button("上一步")
                    tab3_next_button = gr.Button("下一步", variant="primary")

                tab3_prev_button.click(fn=lambda: gr.update(selected="tab2"), outputs=tabs)
                tab3_next_button.click(fn=lambda: gr.update(selected="tab4"), outputs=tabs)

            # TAB 4: Type 筛选日志和 SHAP 图像
            with gr.Tab("Type筛选与SHAP分析", id="tab4"):
                gr.Markdown("## Type 筛选日志与 SHAP 重要性分析")

                with gr.Row():
                    with gr.Column():
                        filter_log_output = gr.Textbox(label="Type 筛选日志", lines=12, max_lines=20, info="显示数据筛选过程和结果")

                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### SHAP 重要性图像选择")
                        shap_type_selector = gr.Dropdown(choices=[], label="选择Type", info="选择要查看的Type的SHAP重要性图像")
                        shap_display_output = gr.Image(label="SHAP 重要性图像")

                with gr.Row():
                    tab4_prev_button = gr.Button("上一步")
                    tab4_next_button = gr.Button("下一步", variant="primary")

                tab4_prev_button.click(fn=lambda: gr.update(selected="tab3"), outputs=tabs)
                tab4_next_button.click(fn=lambda: gr.update(selected="tab5"), outputs=tabs)

            # TAB 5: 最优参数结果
            with gr.Tab("最优参数结果", id="tab5"):
                gr.Markdown("## 各类别最优参数组合结果")

                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### 最优参数结果选择")
                        params_type_selector = gr.Dropdown(choices=[], label="选择Type", info="选择要查看的Type的最优参数结果")
                        params_display_output = gr.Dataframe(label="最优参数结果表格")

                with gr.Row():
                    tab5_prev_button = gr.Button("上一步")

                tab5_prev_button.click(fn=lambda: gr.update(selected="tab4"), outputs=tabs)

            # 运行按钮事件
            filter_log_state = gr.State("")  # 筛选日志状态
            shap_images_state = gr.State({})  # SHAP图像状态
            best_params_state = gr.State({})  # 最优参数状态
            rmse_image_state = gr.State(None)  # RMSE图像状态

            run_button.click(
                fn=run_system,
                inputs=[excel_file_output, sence_columns, process_columns, type_columns, result_columns, init_vector, filter_threshold, search_threshold, search_num_iter, search_multiplier, search_candidates_per_round, search_target_number, select_threshold, select_max_num],
                outputs=[status_output, rmse_image_state, filter_log_state, shap_images_state, best_params_state],
            )

            # 更新选择器选项和默认图像/表格
            def update_shap_selector(shap_images):
                if not shap_images:
                    return gr.update(choices=[], value=None), gr.update(value=None)
                types = list(shap_images.keys())
                return gr.update(choices=types, value=types[0] if types else None), shap_images[types[0]] if types else None

            def update_shap_image(type_name, shap_images):
                if not type_name or type_name not in shap_images:
                    return None
                return shap_images[type_name]

            def update_params_selector(best_params):
                if not best_params:
                    return gr.update(choices=[], value=None), gr.update(value=None)
                types = list(best_params.keys())
                if types and best_params[types[0]]:
                    # 将字典列表转换为DataFrame可以显示的格式
                    first_type_data = best_params[types[0]]
                    if first_type_data:
                        # 获取表头
                        headers = list(first_type_data[0].keys()) if isinstance(first_type_data[0], dict) else []
                        # 构建二维列表数据
                        data = [[row[header] for header in headers] for row in first_type_data] if headers else first_type_data
                        # 返回包含headers和data的字典
                        return gr.update(choices=types, value=types[0] if types else None), {"headers": headers, "data": data}
                return gr.update(choices=types, value=types[0] if types else None), None

            def update_params_table(type_name, best_params):
                if not type_name or type_name not in best_params:
                    return None
                type_data = best_params[type_name]
                if not type_data:
                    return None
                # 将字典列表转换为DataFrame可以显示的格式
                if isinstance(type_data[0], dict):
                    # 获取表头
                    headers = list(type_data[0].keys())
                    # 构建二维列表数据
                    data = [[row[header] for header in headers] for row in type_data]
                    # 返回包含headers和data的字典
                    return {"headers": headers, "data": data}
                return type_data

            # 监听状态变化，更新界面内容
            filter_log_state.change(fn=lambda log: gr.update(value=log), inputs=filter_log_state, outputs=filter_log_output)

            # 更新SHAP图像选择器和默认图像
            shap_images_state.change(fn=update_shap_selector, inputs=shap_images_state, outputs=[shap_type_selector, shap_display_output])

            # 根据选择器更新SHAP图像
            shap_type_selector.change(fn=update_shap_image, inputs=[shap_type_selector, shap_images_state], outputs=shap_display_output)

            # 更新最优参数选择器和默认表格
            best_params_state.change(fn=update_params_selector, inputs=best_params_state, outputs=[params_type_selector, params_display_output])

            # 根据选择器更新最优参数表格
            params_type_selector.change(fn=update_params_table, inputs=[params_type_selector, best_params_state], outputs=params_display_output)

            # 移除自动跳转，用户可通过下一步按钮手动切换

    return demo


if __name__ == "__main__":
    demo = ui()
    demo.launch()
