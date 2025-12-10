import os
from pathlib import Path

def get_temp_path():
        # 1. 获取当前脚本所在的目录
    script_dir = Path(__file__).parent  # __file__ 是当前脚本文件路径

    # 2. 获取上级目录（即 ../）
    parent_dir = script_dir.parent

    # 3. 拼接目标相对路径：../temp/ → 即 parent_dir / "temp"
    target_dir = parent_dir / "temp"

    target_dir.mkdir(parents=True, exist_ok=True)  # 创建目录（如果不存在）

    # 4. （可选）转换为绝对路径
    abs_target_path = target_dir.resolve()  # 解析为绝对路径，更清晰

    return abs_target_path

def get_temp_image_path():
    target_dir = get_temp_path() / "images"

    target_dir.mkdir(parents=True, exist_ok=True)  # 创建目录（如果不存在）

    # 4. （可选）转换为绝对路径
    abs_target_path = target_dir.resolve()  # 解析为绝对路径，更清晰

    return abs_target_path

def get_temp_csv_path():
    target_dir = get_temp_path() / "csv"

    target_dir.mkdir(parents=True, exist_ok=True)  # 创建目录（如果不存在）

    # 4. （可选）转换为绝对路径
    abs_target_path = target_dir.resolve()  # 解析为绝对路径，更清晰

    return abs_target_path
