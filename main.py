import re
from matplotlib.pylab import f
import shap
from configs import Config
import pandas as pd
from sklearn.preprocessing import StandardScaler
from modules.data_filter import data_filter
from modules.regression import find_best_model, RegressionModel, plot_multi_types
from modules.select_matrix import SelectMatrix


def main():
    config = Config()
    filepath = config.get("datasets.path")
    sence_columns = config.get("datasets.sence_columns")
    process_columns = config.get("datasets.process_columns")
    type_columns = config.get("datasets.type_columns")
    result_columns = config.get("datasets.result_columns")
    df = pd.read_excel(filepath)

    df = data_filter(df, sence_columns, config.get("init_data"), config.get("filter_threshold"))
    types = sorted(df["Type"].unique().tolist())

    type_models = {}
    type_results = {}
    pre_select = False
    modified = True

    # test
    selectMatrix = SelectMatrix(
        sence_columns,
        process_columns,
        types,
        [
            [1, 1, 1, 0, 1, 1, 0, 0],
            [1, 1, 1, 1, 0, 0, 1, 0],
            [0, 1, 1, 1, 1, 0, 0, 1],
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

            regression_model.plot_shap_importance()

        plot_multi_types(type_results)
        pre_select = True
        modified = selectMatrix.interactive_edit()


if __name__ == "__main__":
    main()
