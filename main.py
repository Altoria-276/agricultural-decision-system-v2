import re
from matplotlib.pylab import f
import shap
from configs import Config
import pandas as pd
from sklearn.preprocessing import StandardScaler
from modules.data_filter import data_filter
from modules.regression import find_best_model, RegressionModel, plot_multi_types
from modules.select_matrix import SelectMatrix

from modules.get_best_type import get_best_type
from modules.search_parameters import search_parameters

def main():
    config = Config()
    filepath = config.get("datasets.path")
    sence_columns = config.get("datasets.sence_columns")
    process_columns = config.get("datasets.process_columns")
    type_columns = config.get("datasets.type_columns")
    result_columns = config.get("datasets.result_columns")
    init_data = config.get("init_data") # 用户输入数据
    df = pd.read_excel(filepath)

    df = data_filter(df, sence_columns, init_data, config.get("filter_threshold")) # 筛选相似数据

    # 数量检测，删去数据量少于 3 的 type
    type_counts = df["Type"].value_counts()
    print("各 Type 的数据量：")
    for t, cnt in type_counts.items():
        print(f"  - {t}: {cnt} 条")
    invalid_types = type_counts[type_counts < 3].index.tolist()
    if invalid_types:
        print("以下 Type 数据不足 3 条，将被删除：", invalid_types)
    else:
        print("所有 Type 均满足数据量要求")
    df = df.groupby("Type").filter(lambda x: len(x) >= 3)
    # 更新 types 列表
    types = sorted(df["Type"].unique().tolist())
    print("过滤后的有效 Type:", types)


    type_models = {}
    type_results = {}
    pre_select = False
    modified = True

    best_models = {} # 保存各类别最优模型

    # test
    selectMatrix = SelectMatrix(
        sence_columns,
        process_columns,
        types,
        [
            [1, 1, 1, 0, 0, 1, 1, 0],
            [0, 1, 1, 1, 0, 0, 1, 1],
            [0, 0, 1, 1, 1, 1, 0, 1],
        ][: len(types)],
    )

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
            best_models[type] = regression_model # 保存各类别最优模型
            #regression_model.plot_shap_importance()

        #plot_multi_types(type_results)
        pre_select = True
        modified = selectMatrix.interactive_edit()

    # 选择最优模型
    best_type = get_best_type(best_models, types, init_data, df)
    print(f"最优的类型是: {best_type}")
    
    # 搜索最优参数
    df_best_type = df[df["Type"] == best_type].copy() # 筛选最优类型数据
    best_params_and_results = search_parameters(best_models[best_type], init_data, df_best_type, config, config.get("search_params_threshold"), config.get("search_params_num_iter"))
    print(f"最优的参数组合及结果是:\n {best_params_and_results.to_string(index=False)}")
    
    

if __name__ == "__main__":
    main()
