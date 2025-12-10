#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试 SelectMatrix 类的新方法：load_or_create_matrix_csv、save_matrix_csv 和 update_matrix_from_csv
"""

import os
import sys
import pandas as pd

# 添加项目根目录到 Python 路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from modules.select_matrix import SelectMatrix


def test_select_matrix_csv():
    """测试选择矩阵的CSV文件处理功能"""
    print("=" * 60)
    print("测试 SelectMatrix 的 CSV 文件处理功能")
    print("=" * 60)

    # 测试数据
    sence_columns = ["温度", "湿度", "光照"]
    process_columns = ["施肥量", "灌溉量"]
    types = ["type1", "type2", "type3"]
    selected_columns = ["温度", "施肥量"]

    # 创建 SelectMatrix 实例
    select_matrix = SelectMatrix(sence_columns=sence_columns, process_columns=process_columns, types=types, selected_columns=selected_columns)

    print("\n1. 初始矩阵:")
    print(select_matrix.matrix_df)

    # 测试文件路径
    test_file_path = "data/test_select_matrix.csv"

    # 测试 1: 加载或创建矩阵 CSV
    print("\n2. 测试 load_or_create_matrix_csv:")
    result = select_matrix.load_or_create_matrix_csv(test_file_path)
    print(f"   结果: {'成功' if result else '失败'}")

    # 测试 2: 读取并验证创建的文件
    print("\n3. 验证创建的 CSV 文件:")
    if os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)), test_file_path)):
        df = pd.read_csv(test_file_path, index_col=0, encoding="utf-8")
        print(df)
    else:
        print("   ❌ CSV 文件未创建成功")

    # 测试 3: 修改矩阵并保存
    print("\n4. 测试 save_matrix_csv:")
    # 修改矩阵，确保所有类型的选中列数都为 3
    select_matrix["type1"] = ["温度", "湿度", "施肥量"]
    select_matrix["type2"] = ["湿度", "光照", "灌溉量"]
    select_matrix["type3"] = ["温度", "光照", "施肥量"]
    print("   修改后的矩阵:")
    print(select_matrix.matrix_df)
    # 保存到 CSV
    result = select_matrix.save_matrix_csv(test_file_path)
    print(f"   保存结果: {'成功' if result else '失败'}")

    # 测试 4: 从 CSV 更新矩阵
    print("\n5. 测试 update_matrix_from_csv:")
    # 创建一个新的 SelectMatrix 实例，使用与保存时相同的 select_nums
    new_select_matrix = SelectMatrix(
        sence_columns=sence_columns,
        process_columns=process_columns,
        types=types,
        select_nums=3,  # 使用与修改后矩阵相匹配的 select_nums
    )
    print("   新实例的初始矩阵:")
    print(new_select_matrix.matrix_df)
    # 从 CSV 更新
    result = new_select_matrix.update_matrix_from_csv(test_file_path)
    print(f"   更新结果: {'成功' if result else '失败'}")
    print("   更新后的矩阵:")
    print(new_select_matrix.matrix_df)

    # 测试 5: 测试非法 CSV 文件的验证
    print("\n6. 测试非法 CSV 文件的验证:")
    # 创建一个非法的 CSV 文件
    illegal_df = pd.DataFrame([[1, 2, 0, 1, 0], [1, 0, 0, 1, 0], [1, 0, 0, 1, 0]], index=types, columns=sence_columns + process_columns)
    illegal_df.to_csv(test_file_path, index=True, encoding="utf-8")
    # 尝试从非法文件更新
    result = new_select_matrix.update_matrix_from_csv(test_file_path)
    print(f"   验证结果: {'失败（预期行为）' if not result else '通过（非预期行为）'}")

    # 清理测试文件
    print("\n7. 清理测试文件:")
    test_file_abs_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), test_file_path)
    if os.path.exists(test_file_abs_path):
        os.remove(test_file_abs_path)
        print(f"   ✅ 已删除测试文件: {test_file_path}")
    else:
        print(f"   ℹ️  测试文件不存在，无需删除")

    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)


if __name__ == "__main__":
    test_select_matrix_csv()
