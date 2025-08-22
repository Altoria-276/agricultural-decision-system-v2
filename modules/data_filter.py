from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd


def data_filter(df, columns, vector, threshold):
    # 1. 检查DataFrame列数与向量长度是否一致
    if df[columns].shape[1] != len(vector):
        raise ValueError("DataFrame columns length does not match vector length")

    # 2. 初始化StandardScaler并对数据进行归一化
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df[columns])

    # 将向量转换为二维数组并进行归一化
    vector_reshaped = np.array(vector).reshape(1, -1)
    vector_scaled = scaler.transform(vector_reshaped)

    # 3. 计算余弦相似度并筛选数据
    similarities = cosine_similarity(df_scaled, vector_scaled).flatten()
    mask = similarities > threshold
    filtered_scaled = df_scaled[mask]

    # 4. 对筛选后的数据进行反归一化
    filtered_data = scaler.inverse_transform(filtered_scaled)

    # 创建结果DataFrame，保留原始数据的索引和其他列
    result_df = df.loc[mask].copy()
    result_df[columns] = filtered_data

    return result_df
