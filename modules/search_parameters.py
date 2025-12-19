from configs.config import Config
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
    regression_model: RegressionModel,
    input_data,
    filtered_dataset: pd.DataFrame,
    config: Config,
    target_y: float,
    threshold: float,
    multiplier: float,
    num_iter: int,
    num_candidates_per_round: int,
):
    """
    搜索参数(此时已知 type)

    Args:
        regression_model: 对应 type 的 RegressionModel 实例
        input_data: 输入向量 (list[float])，长度 = sence_columns 数量
        filtered_dataset: 数据集(只包含此 type 的数据)
        config: 配置文件字典
        target_y: 用户输入的目标值
        threshold: 预设阈值
        multiplier: 目标值乘数
        num_iter: 迭代次数
        num_candidates_per_round: 每轮选择的最优数据数量

    Returns:
        pd.DataFrame: 输出满足条件的参数数据
    """
    # 1. 用 input_data 覆盖环境因子列
    sence_cols = config.get("datasets.sence_columns")
    process_cols = config.get("datasets.process_columns")
    target_value = target_y * multiplier

    filtered_dataset = filtered_dataset.copy()

    filtered_dataset.loc[:, sence_cols] = input_data

    # 2. 确定可拓展的列（工艺参数和模型特征的交集）
    target_cols = [col for col in regression_model.feature if col in process_cols]
    if not target_cols:
        raise ValueError("模型选择的特征与 process_columns 无交集，无法拓展")
    # print(f"将要拓展的列：{target_cols}")
    current_dataset = filtered_dataset.copy()

    for i in range(num_iter):
        # 3. 拓展数据集（在工艺参数上做扩展）
        expanded_dataset = expand_dataset(current_dataset, target_cols)
        expanded_dataset = expanded_dataset.drop_duplicates(subset=process_cols, keep="first")

        # 4. 预测
        predictions = regression_model.predict_inverse_transform(expanded_dataset)

        # 5. 选择最接近 target_value 的 num_candidates_per_round 条数据用于下一轮迭代
        target_dim = len(target_cols)
        desired_k = 2 if target_dim == 1 else num_candidates_per_round
        top_idx = np.argsort(np.abs(predictions - target_value))[:desired_k]
        current_dataset = expanded_dataset.iloc[top_idx].copy()
        current_predictions = predictions[top_idx]

    # 6. 最后一轮筛选
    mask = current_predictions > threshold
    # mask = current_predictions < target_y
    output_filtered = current_dataset.loc[mask, target_cols].copy()

    pred_col_name = regression_model.target[0]
    output_filtered[pred_col_name] = current_predictions[mask]

    return output_filtered.reset_index(drop=True)
