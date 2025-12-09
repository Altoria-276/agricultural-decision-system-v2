import numpy as np


def select_variables(correlation_matrix, shapley_values, types, variable_names, m, threshold_t):
    """
    指标分析方法，从给定的N个变量中筛选出m个变量
    Args:
        correlation_matrix: 原始N个变量之间的相关性矩阵
        shapley_values: 原始N个变量的shapley值列表
        types: 原始N个变量的类型列表，S or P
        variable_names: 原始N个变量的变量名 # todo 可选？
        m: 需要筛选出的变量数量
        threshold_t: 强相关性阈值

    Returns:
        selected_corr_matrix: 筛选出的m个变量的相关性矩阵
        selected_variable_names: 筛选出的m个变量的变量名 # todo 可选？
    """
    # 将 list 转为 numpy 数组
    shapley_values = np.array(shapley_values)

    if shapley_values.ndim == 2:
        shapley_values = np.nanmean(np.abs(shapley_values), axis=0)

    # Step 1: 找出 S 和 P 中 Shapley 值最大的索引
    s_indices = [i for i, t in enumerate(types) if t == "S"]
    p_indices = [i for i, t in enumerate(types) if t == "P"]

    if not s_indices or not p_indices:
        raise ValueError("Input must contain at least one 'S' and one 'P' type variable.")

    max_s_index = s_indices[np.argmax([shapley_values[i] for i in s_indices])]
    max_p_index = p_indices[np.argmax([shapley_values[i] for i in p_indices])]

    selected_indices = []

    if shapley_values[max_s_index] > shapley_values[max_p_index]:
        selected_indices = [max_s_index, max_p_index]
    else:
        selected_indices = [max_p_index, max_s_index]

    # Step 2: 剩余变量按 Shapley 值降序排列（排除已选的两个）

    remaining_indices = [i for i in np.argsort(-shapley_values) if i not in selected_indices]

    # Step 3: 逐步筛选剩余变量
    for idx in remaining_indices:
        if len(selected_indices) >= m or shapley_values[idx] == 0:
            break
        valid = True

        for sel in selected_indices:
            if abs(correlation_matrix.iloc[sel, idx]) > threshold_t:
                valid = False
                break
        if valid:
            selected_indices.append(idx)

    # Step 4: 若仍不足 m 个变量，从剩余未选中的元素中继续添加
    unused_indices = [idx for idx in remaining_indices if idx not in selected_indices]
    while len(selected_indices) < m and unused_indices:
        selected_indices.append(unused_indices.pop(0))

    # Step 5: 构建输出
    selected_corr_matrix = correlation_matrix.iloc[selected_indices, selected_indices]
    selected_variable_names = [variable_names[i] for i in selected_indices]

    return selected_corr_matrix, selected_variable_names
