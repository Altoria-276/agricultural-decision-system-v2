import pandas as pd
import numpy as np


class SelectMatrix:

    def __init__(self, sence_columns: list[str], process_columns: list[str], types: list[str], data: list[list[int]]):

        self.sence_columns = sence_columns
        self.process_columns = process_columns
        self.types = types

        # 检查 data 的维度是否匹配
        expected_columns = len(sence_columns) + len(process_columns)
        if len(data) != len(types):
            raise ValueError(f"data 的行数 ({len(data)}) 与 types 的数量 ({len(types)}) 不匹配")
        for row in data:
            if len(row) != expected_columns:
                raise ValueError(
                    f"data 的每一行应有 {expected_columns} 列（{len(sence_columns) + len(process_columns)} 列），但发现为 {len(row)} 列"
                )

        self.matrix_df = pd.DataFrame(np.array(data), index=types, columns=self.sence_columns + self.process_columns)

    def __getitem__(self, type_key: str) -> list[str]:
        """
        重载 [] 操作符，使得 selectMatrix[type] 返回该 type 对应选中（值为1）的列名列表。

        参数:
            type_key (str): 某个 type 名称，如 "type1"

        返回:
            list[str]: 该 type 对应行中值为 1 的列名列表
        """
        if type_key not in self.types:
            raise KeyError(f"Type '{type_key}' 不存在于 types 列表中。可用 types: {self.types}")

        # 获取该 type 对应的行，筛选出值为 1 的列
        selected_columns = self.matrix_df.loc[type_key][self.matrix_df.loc[type_key] == 1].index.tolist()
        return selected_columns

    def interactive_edit(self) -> bool:
        """
        交互式命令行编辑选择矩阵。

        支持用户输入如 "A 0 0" 来修改 type A 的第 0 列的值为 0。
        输入 -1 退出编辑。

        返回:
            bool: 是否修改了矩阵的任何值（True/False）
        """
        modified = False  # 标记是否有修改发生

        while True:
            # 打印当前矩阵
            print("\n当前选择矩阵：")
            print(self.matrix_df)
            print("\n当前 Type 列表:", self.types)
            print("当前所有列:", self.matrix_df.columns.tolist())
            print("\n请输入修改指令，格式如：A 0 0 （将 Type A 的第 0 列设置为 0）")
            print("输入 -1 退出编辑")

            user_input = input(">>> ").strip()

            # 退出条件
            if user_input == "-1":
                print("退出编辑。")
                break

            parts = user_input.split()
            if len(parts) != 3:
                print(f"❌ 输入格式错误！请输入：Type 列索引 新值（如 'A 0 1'）。你输入的是：{user_input}")
                continue

            type_str, col_idx_str, new_value_str = parts

            # 检查 type 是否存在
            if type_str not in self.types:
                print(f"❌ Type '{type_str}' 不存在！可用 Type: {self.types}")
                continue

            # 检查列索引是否为整数且在范围内
            try:
                col_idx = int(col_idx_str)
            except ValueError:
                print(f"❌ 列索引 '{col_idx_str}' 不是有效的整数！")
                continue

            if col_idx < 0 or col_idx >= len(self.matrix_df.columns):
                print(f"❌ 列索引 {col_idx} 超出范围！有效范围：0 ~ {len(self.matrix_df.columns) - 1}")
                continue

            # 检查新值是否为 0 或 1
            if new_value_str not in ["0", "1"]:
                print(f"❌ 新值 '{new_value_str}' 必须是 0 或 1！")
                continue

            new_value = int(new_value_str)

            # 获取当前值
            current_val = self.matrix_df.at[type_str, self.matrix_df.columns[col_idx]]

            # 如果当前值等于新值，则无需修改
            if current_val == new_value:
                print(f"ℹ️  Type '{type_str}' 的第 {col_idx} 列已经是 {new_value}，未修改。")
            else:
                # 修改值
                self.matrix_df.at[type_str, self.matrix_df.columns[col_idx]] = new_value
                print(f"✅ 已修改：Type '{type_str}' 的第 {col_idx} 列 -> {new_value}")
                modified = True

        return modified
