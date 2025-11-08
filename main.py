import re
from matplotlib.pylab import f
from shap import Explanation
from configs import Config
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from modules.data_filter import data_filter
from modules.regression import find_best_model, RegressionModel, plot_multi_types
from modules.select_matrix import SelectMatrix
from modules.get_best_type import get_best_type
from modules.search_parameters import search_parameters
from modules.indicator_analysis import select_variables

import warnings

warnings.filterwarnings("ignore")


def main():
    config = Config()
    filepath = config.get("datasets.path")
    sence_columns = config.get("datasets.sence_columns")
    process_columns = config.get("datasets.process_columns")
    type_columns = config.get("datasets.type_columns")
    result_columns = config.get("datasets.result_columns")
    init_data = config.get("init_data")  # 用户输入数据
    df = pd.read_excel(filepath)

    df = data_filter(df, sence_columns, init_data, config.get("filter_threshold"))  # 筛选相似数据

    # 数量检测，删去数据量少于 3 的 type
    type_counts = df[type_columns[0]].value_counts()
    print("各 Type 的数据量：")
    for t, cnt in type_counts.items():
        print(f"  - {t}: {cnt} 条")
    invalid_types = type_counts[type_counts < 3].index.tolist()
    if invalid_types:
        print("以下 Type 数据不足 3 条，将被删除：", invalid_types)
    else:
        print("所有 Type 均满足数据量要求")
    df = df.groupby(type_columns[0]).filter(lambda x: len(x) >= 3)
    # 更新 types 列表
    types = sorted(df[type_columns[0]].unique().tolist())
    print("过滤后的有效 Type:", types)

    type_models = {}
    type_results = {}
    pre_select = False
    modified = True

    best_models = {}  # 保存各类别最优模型

    # test
    # selectMatrix = SelectMatrix(
    #     sence_columns,
    #     process_columns,
    #     types,
    #     [
    #         [1, 1, 1, 0, 0, 1, 1, 0],
    #         [0, 1, 1, 1, 0, 0, 1, 1],
    #         [0, 0, 1, 1, 1, 1, 0, 1],
    #     ][: len(types)],
    # )

    while modified:
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

                print(f"类型: {type}")
                print(results)
                print(f"最好的模型是: {model_select}")

                # STEP 3
                # TODO

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
                    m=config.get("select_variables_max_num"),  #
                    threshold_t=config.get("select_variables_threshold"),
                )

                selectMatrix = SelectMatrix(
                    sence_columns,
                    process_columns,
                    types,
                    selected_columns=selected_feature_names,
                )

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
            regression_model.plot_shap_importance()

        plot_multi_types(type_results)
        pre_select = True
        modified = selectMatrix.interactive_edit()

    # v2 UPDATE
    # 最优模型选择被注释，对每个类型均进行最优参数搜索

    # 选择最优模型
    # best_type = get_best_type(best_models, types, type_columns, init_data, df)
    # print(f"最优的类型是: {best_type}")

    # 获取用户输入 Y
    while True:
        try:
            target_y_str = input("请输入目标值 Y: ")
            target_y = float(target_y_str)
            break
        except ValueError:
            print("无效输入，请输入一个数字。")

    # 搜索最优参数
    for type in types:
        df_best_type = df[df[type_columns[0]] == type].copy()  # 筛选最优类型数据
        
        # 从配置文件中获取 search_parameters 相关参数
        search_config = config.get("search_parameters")
        threshold = search_config.get("threshold", 10)
        multiplier = search_config.get("multiplier", 0.95)
        num_iter = search_config.get("num_iter", 4)
        num_candidates_per_round = search_config.get("num_candidates_per_round", 5)
        
        best_params_and_results = search_parameters(
            regression_model=best_models[type], 
            input_data=init_data, 
            filtered_dataset=df_best_type, 
            config=config, 
            target_y=target_y,
            threshold=threshold,
            multiplier=multiplier,
            num_iter=num_iter,
            num_candidates_per_round=num_candidates_per_round
        )
        print(f"类型 {type} 的最优参数组合及结果是:\n {best_params_and_results.to_string(index=False)}")


if __name__ == "__main__":
    main()
