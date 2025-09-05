import pandas as pd
from modules.regression import RegressionModel


def get_best_type(
    models: dict, types: list[str], type_columns: list[str], input_data: list[float], filtered_dataset: pd.DataFrame
) -> str | None:
    """
    根据用户输入和筛选后的数据，选择最优类型。
    每个类型直接使用其对应模型和特征进行预测，取均值作为得分。

    Args:
        models (dict[str, RegressionModel]): 各类型对应的 RegressionModel 实例，key=type
        types (list[str]): 所有类型
        type_columns (pd.DataFrame): 类型列名
        input_data (list[float]): 用户输入数据
        filtered_dataset (pd.DataFrame): 经过筛选的相似数据，包含 type 列

    Returns:
        str | None: 最优类型，若无可选类型返回 None
    """

    # 1. 拼接数据：用 input_data 覆盖 filtered_dataset 前 len(input_data) 列
    filtered_dataset = filtered_dataset.copy()
    feature_cols = filtered_dataset.columns[: len(input_data)]
    filtered_dataset.loc[:, feature_cols] = input_data

    # 2. 按类型划分
    scores = {}
    for t in types:
        subset = filtered_dataset[filtered_dataset[type_columns[0]] == t]
        # 直接用模型预测（内部会自动选择所需特征）
        score = models[t].predict_inverse_transform(subset).mean()
        scores[t] = score

    # 3. 返回得分最高的类型
    if not scores:
        return None
    return max(scores, key=scores.get)
