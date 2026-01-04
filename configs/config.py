import yaml
import os
from pathlib import Path


class Config:
    """YAML 配置文件读取工具类，自动忽略与 default.yaml 相同的配置项"""

    DEFAULT_CONFIG_PATH = Path("configs/default.yaml")

    def __init__(self, config_path: str = "configs/custom.yaml"):
        self.config_path = Path(config_path)
        self.default_config = self._load_config_file(self.DEFAULT_CONFIG_PATH)
        self.config = self._load_config_file(self.config_path, self.default_config)

    def _load_config_file(self, path: Path, defaults: dict = None) -> dict:
        """加载配置文件，支持默认配置覆盖"""
        if not path.exists():
            return defaults or {}

        try:
            with open(path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f) or {}
                return self._merge_configs(defaults or {}, config) if defaults else config
        except Exception as e:
            raise ValueError(f"配置文件解析错误: {e}")

    def _merge_configs(self, defaults: dict, user_config: dict) -> dict:
        """递归合并配置（用户配置优先）"""
        merged = defaults.copy()
        for key, value in user_config.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key] = self._merge_configs(merged[key], value)
            else:
                merged[key] = value
        return merged

    def _get_nested(self, key: str):
        """使用点符号获取嵌套配置值（支持环境变量）"""
        value = self.config
        for k in key.split("."):
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                raise KeyError(f"配置键 '{key}' 不存在")

        if isinstance(value, str) and value.startswith("env:"):
            return os.environ.get(value[4:], value)
        return value

    def __getitem__(self, key: str):
        return self.get(key)

    def __setitem__(self, key: str, value):
        self.update(key, value)

    def update(self, key: str, value):
        """更新配置值"""
        self._set_nested(key, value)
        self._save()  # 保存时自动移除相同配置

    def _set_nested(self, key: str, value):
        """使用点符号设置嵌套配置值"""
        keys = key.split(".")
        current = self.config
        for k in keys[:-1]:
            current = current.setdefault(k, {})
        current[keys[-1]] = value

    def _save(self):
        """保存配置（仅保存与 default.yaml 差异的配置项）"""
        # 计算差异配置（只保留与默认不同的部分）
        diff_config = self._get_diff_config(self.config, self.default_config)

        # 如果没有差异，删除配置文件
        if not diff_config:
            if self.config_path.exists():
                os.remove(self.config_path)
            return

        # 保存差异配置
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.config_path, "w", encoding="utf-8") as f:
            yaml.dump(diff_config, f, default_flow_style=False, allow_unicode=True)

    def _get_diff_config(self, config: dict, defaults: dict) -> dict:
        """递归获取与默认配置的差异（仅保留不同部分）"""
        diff = {}
        for key, value in config.items():
            # 如果默认配置中没有该键，直接添加
            if key not in defaults:
                diff[key] = value
            # 如果值是字典，递归比较
            elif isinstance(value, dict) and isinstance(defaults[key], dict):
                nested_diff = self._get_diff_config(value, defaults[key])
                if nested_diff:  # 仅当嵌套部分有差异时才添加
                    diff[key] = nested_diff
            # 普通值比较（跳过相同值）
            elif value != defaults[key]:
                diff[key] = value
        return diff

    def reload(self):
        """重新加载配置文件（自动移除相同配置）"""
        self.config = self._load_config_file(self.config_path, self.default_config)

    def to_dict(self):
        """返回所有配置的字典形式（包含所有配置项，包括默认项）"""
        return self.config.copy()

    def clear(self):
        """清空 custom.yaml（自动删除文件，因为无差异）"""
        self.config = {}
        self._save()
        self.reload()

    def get(self, key: str):
        return self._get_nested(key)
