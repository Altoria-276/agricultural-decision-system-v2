import pandas as pd
import numpy as np
from modules.regression import RegressionModel
from itertools import product

def expand_dataset(dataset, target_cols, quantile_levels=[0.05,0.5,0.95]):
    # 收集每列的分位值
    col_values = [dataset[col].quantile(quantile_levels).values for col in target_cols]
    # 全排列
    combos = list(product(*col_values))
    # 用 combos 构建 DataFrame
    expanded = pd.DataFrame(combos, columns=target_cols)
    # 复制 dataset 的其他列
    non_target_cols = [c for c in dataset.columns if c not in target_cols]
    for col in non_target_cols:
        expanded[col] = dataset.iloc[0][col]
    return expanded


def search_parameters(
    regression_model, 
    input_data: list[float],
    filtered_dataset: pd.DataFrame, 
    config: dict, 
    threshold: float, 
    num_iter: int = 4
):
    """
    搜索参数(此时已知 type)

    Args:
        regression_model: 对应 type 的 RegressionModel 实例
        input_data: 输入向量 (list[float])，长度 = sence_columns 数量
        filtered_dataset: 数据集(只包含此 type 的数据)
        config: 配置文件字典
        threshold: 预设阈值
        num_iter: 迭代次数

    Returns:
        pd.DataFrame: 输出满足条件的参数数据
    """
    # 1. 用 input_data 覆盖环境因子列
    sence_cols = config.get("datasets.sence_columns")
    process_cols = config.get("datasets.process_columns")

    filtered_dataset = filtered_dataset.copy()
    filtered_dataset.loc[:, sence_cols] = input_data

    # 2. 确定可拓展的列（工艺参数和模型特征的交集）
    target_cols = [col for col in regression_model.feature if col in process_cols]
    if not target_cols:
        raise ValueError("模型选择的特征与 process_columns 无交集，无法拓展")
    print(f"将要拓展的列：{target_cols}")
    current_dataset = filtered_dataset.copy()

    for i in range(num_iter):
        # 3. 拓展数据集（在工艺参数上做扩展）
        expanded_dataset = expand_dataset(current_dataset, target_cols)
        expanded_dataset = expanded_dataset.drop_duplicates(
            subset=process_cols, keep="first"
        )

        # 4. 预测
        predictions = regression_model.predict(expanded_dataset)

        # 5. 选择最高的 5 条数据用于下一轮迭代
        k = min(5, len(predictions))
        top_idx = np.argsort(predictions)[-k:][::-1]
        current_dataset = expanded_dataset.iloc[top_idx].copy()
        current_predictions = predictions[top_idx]

    # 6. 最后一轮筛选
    mask = current_predictions > threshold
    output_filtered = current_dataset.loc[mask, process_cols].copy()
    output_filtered["prediction"] = current_predictions[mask]

    return output_filtered.reset_index(drop=True)
