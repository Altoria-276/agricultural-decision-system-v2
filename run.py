from pathlib import Path
import shutil
from configs.config import Config
import pandas as pd
import numpy as np

from modules.data_filter import data_filter
from modules.regression import find_best_model, RegressionModel, plot_multi_types
from modules.select_matrix import SelectMatrix
from modules.search_parameters import search_parameters
from modules.indicator_analysis import select_variables

import warnings
from utils.filepath import get_temp_path

warnings.filterwarnings("ignore")


def run(config: Config):
    """
    农业决策系统核心运行函数

    Args:
        config: 配置对象，包含所有参数

    Returns:
        dict: 包含运行结果的字典
    """
    try:
        # 删除 temp 目录下的所有文件和子目录
        temp_dir = get_temp_path("temp/")
        if temp_dir.exists():
            for file in temp_dir.iterdir():
                if file.is_file():
                    file.unlink()
                elif file.is_dir():
                    shutil.rmtree(file)

        # 获取配置参数
        filepath = config.get("datasets.path")
        sence_columns = config.get("datasets.sence_columns")
        process_columns = config.get("datasets.process_columns")
        type_columns = config.get("datasets.type_columns")
        result_columns = config.get("datasets.result_columns")
        init_data = config.get("init_data")
        filter_threshold = config.get("filter_threshold")

        select_path: Path = get_temp_path(config.get("output.select_csv_path"))
        rmse_path: Path = get_temp_path(config.get("output.rmse_img_path"))
        shap_path: Path = get_temp_path(config.get("output.shap_img_path"))
        result_path: Path = get_temp_path(config.get("output.result_csv_path"))

        num_k = 5

        # 读取输入数据
        if not Path(filepath).exists():
            raise FileNotFoundError(f"数据文件不存在: {filepath}")

        df = pd.read_excel(filepath)

        # 数据筛选
        df = data_filter(df, sence_columns, init_data, filter_threshold)

        # 收集筛选日志
        filter_log = []
        filter_log.append("各 Type 的数据量：")

        # 数量检测，删去数据量少于 3 的 type
        type_counts = df[type_columns[0]].value_counts()
        print("各 Type 的数据量：")
        for t, cnt in type_counts.items():
            log_line = f"  - {t}: {cnt} 条"
            filter_log.append(log_line)
            print(log_line)
        invalid_types = type_counts[type_counts < num_k].index.tolist()
        if invalid_types:
            log_line = f"以下 Type 数据不足 {num_k} 条，将被删除：{invalid_types}"
            filter_log.append(log_line)
            print(log_line)
        else:
            log_line = "所有 Type 均满足数据量要求"
            filter_log.append(log_line)
            print(log_line)
        df = df.groupby(type_columns[0]).filter(lambda x: len(x) >= num_k)

        # 更新 types 列表
        types = sorted(df[type_columns[0]].unique().tolist())
        log_line = f"过滤后的有效 Type: {types}"
        filter_log.append(log_line)
        print(log_line)

        if len(types) == 0:
            raise ValueError("筛选后没有有效的数据类型")

        type_models = {}
        type_results = {}
        pre_select = False

        best_models = {}  # 保存各类别最优模型
        shap_images = {}  # 保存各类别shap图像路径
        best_params_results = {}  # 保存各类别最优参数结果

        selectMatrix = SelectMatrix(
            sence_columns,
            process_columns,
            types,
            select_nums=config.get("select_variables_max_num"),
        )

        for type in types:
            if not pre_select:
                df_type = df[df[type_columns[0]] == type]

                results, model_select, models = find_best_model(
                    df_type,
                    sence_columns + process_columns,
                    result_columns,
                    config.get("train.test_size"),
                    config.get("train.random_state"),
                )

                type_models[type] = model_select

                print("=" * 50)
                print(f"类型: {type}")
                print(results)
                print(f"最好的模型是: {model_select}")

                corr = models[model_select].correlation_matrix()
                shap_values = models[model_select].shap_importance().values

                importances = np.abs(shap_values).mean(0)
                indices = np.argsort(importances)[::-1]

                sorted_columns = [(sence_columns + process_columns)[i] for i in indices]
                sorted_importances = [importances[i] for i in indices]

                all_feat = sence_columns + process_columns
                sp_types = ["S" if c in sence_columns else "P" for c in all_feat]
                selected_corr_matrix, selected_feature_names = select_variables(
                    correlation_matrix=corr,
                    shapley_values=shap_values,
                    types=sp_types,
                    variable_names=all_feat,
                    m=config.get("select_variables_max_num"),
                    threshold_t=config.get("select_variables_threshold"),
                )

                selectMatrix[type] = selected_feature_names

            selected_colums = selectMatrix[type]
            regression_model = RegressionModel(
                type_models[type],
                df_type,
                selected_colums,
                result_columns,
                config.get("train.test_size"),
                config.get("train.random_state"),
            )

            type_results[type] = regression_model.train_and_evaluate_model()
            best_models[type] = regression_model  # 保存各类别最优模型
            shap_img_name = f"{type}_shap_importance.png"
            regression_model.plot_shap_importance(shap_img_name, shap_path)
            shap_images[type] = str(shap_path / shap_img_name)

        plot_multi_types(type_results, rmse_path)
        pre_select = True

        # 保存每个type选择的列名到CSV文件
        select_columns_path: Path = get_temp_path(select_path) / "select_columns.csv"

        # 创建一个DataFrame来存储每个type选择的列名
        select_columns_data = {}
        for type in types:
            selected_cols = selectMatrix[type]
            select_columns_data[type] = selected_cols

        select_columns_df = pd.DataFrame.from_dict(select_columns_data, orient="index")
        select_columns_df.to_csv(select_columns_path, index=True, encoding="utf-8")

        for type in types:
            df_best_type = df[df[type_columns[0]] == type].copy()  # 筛选最优类型数据

            target_cols = [col for col in best_models[type].feature if col in config.get("datasets.process_columns")]
            uniq_combos = df_best_type[target_cols].drop_duplicates().shape[0] if target_cols else 0
            if uniq_combos < 2:
                print(f"[跳过] Type={type} 在 target_cols={target_cols} 上仅有 {uniq_combos} 种组合，至少需要 2 种才能进行 search_parameters。")
                continue

            # 从配置文件中获取 search_parameters 相关参数
            search_config = config.get("search_parameters")
            threshold = search_config.get("threshold", 10)
            multiplier = search_config.get("multiplier", 0.95)
            num_iter = search_config.get("num_iter", 4)
            num_candidates_per_round = search_config.get("num_candidates_per_round", 5)
            target_y = search_config.get("target_number", 0.5)

            best_params_and_results = search_parameters(
                regression_model=best_models[type],
                input_data=init_data,
                filtered_dataset=df_best_type,
                config=config,
                target_y=target_y,
                threshold=threshold,
                multiplier=multiplier,
                num_iter=num_iter,
                num_candidates_per_round=num_candidates_per_round,
            )
            # 保存最优参数结果
            best_params_results[type] = best_params_and_results

            search_result = "没有符合条件的结果" if best_params_and_results.empty else best_params_and_results.to_string(index=False)
            print(f"类型 {type} ，最优参数组合及结果是:\n {search_result}")
            print("=" * 50)

            csv_path = result_path / f"{type}_best_params.csv"
            best_params_and_results.to_csv(csv_path, index=False)

        # 返回结果
        rmse_image_path = rmse_path / "multi_types_rmse.png"

        # 转换最优参数结果为字典格式，方便前端展示
        best_params_dict = {}
        for type_name, params_df in best_params_results.items():
            if not params_df.empty:
                best_params_dict[type_name] = params_df.to_dict(orient="records")
            else:
                best_params_dict[type_name] = []

        return {
            "success": True,
            "message": "系统运行成功",
            "rmse_image": str(rmse_image_path) if rmse_image_path.exists() else None,
            "types_processed": types,
            "data_summary": {"total_types": len(types), "original_data_shape": df.shape, "filtered_data_shape": df.shape},
            "filter_log": "\n".join(filter_log),
            "shap_images": shap_images,
            "best_params_results": best_params_dict,
        }

    except Exception as e:
        return {"success": False, "message": f"系统运行失败: {str(e)}", "rmse_image": None, "types_processed": [], "data_summary": {}, "filter_log": "", "shap_images": {}, "best_params_results": {}}
