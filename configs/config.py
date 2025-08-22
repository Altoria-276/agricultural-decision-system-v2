import yaml
import os
from pathlib import Path


class Config:
    """YAML 配置文件读取工具类，使用 default.yaml 作为默认配置"""

    # 固定的默认配置文件路径
    DEFAULT_CONFIG_PATH = Path("configs/default.yaml")

    def __init__(self, config_path: str = "configs/custom.yaml"):
        """
        初始化配置加载器

        Args:
            config_path: 用户配置文件路径，默认为 config.yaml
        """
        self.config_path = Path(config_path)
        self.default_config = self._load_default_config()
        self.config = self._load_config()

    def _load_default_config(self) -> dict[str]:
        """加载默认配置文件"""
        if not self.DEFAULT_CONFIG_PATH.exists():
            # 如果默认配置文件不存在，返回空字典
            print(f"提示: 默认配置文件 {self.DEFAULT_CONFIG_PATH} 不存在")
            return {}

        try:
            with open(self.DEFAULT_CONFIG_PATH, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            raise ValueError(f"默认配置文件解析错误: {e}")

    def _load_config(self) -> dict[str]:
        """加载并解析用户配置文件，覆盖默认配置"""
        if not self.config_path.exists():
            # 如果用户配置文件不存在，使用默认配置
            print(f"提示: 配置文件 {self.config_path} 不存在，使用默认配置")
            return self.default_config.copy()

        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                user_config = yaml.safe_load(f) or {}
        except Exception as e:
            raise ValueError(f"配置文件解析错误: {e}")

        # 合并默认配置和用户配置（用户配置优先）
        return self._merge_configs(self.default_config, user_config)

    def _merge_configs(self, defaults: dict[str], user_config: dict[str]) -> dict[str]:
        """递归合并默认配置和用户配置，用户配置优先"""
        merged = defaults.copy()

        for key, value in user_config.items():
            # 如果键在默认配置中存在且都是字典，则递归合并
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key] = self._merge_configs(merged[key], value)
            else:
                # 否则直接使用用户配置文件中的值（覆盖默认值）
                merged[key] = value

        return merged

    def get(self, key: str = None):
        """
        获取配置值，支持点符号访问嵌套键

        Args:
            key: 配置键名，支持点符号如 'database.host'
            default: 如果键不存在时返回的默认值

        Returns:
            配置值或默认值
        """
        try:
            return self._get_nested(key)
        except (KeyError, TypeError):
            return None

    def _get_nested(self, key: str):
        """使用点符号获取嵌套配置值"""
        keys = key.split(".")
        value = self.config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                raise KeyError(f"配置键 '{key}' 不存在")

        # 检查是否需要从环境变量中获取值
        if isinstance(value, str) and value.startswith("env:"):
            env_var = value[4:]
            return os.environ.get(env_var, value)

        return value

    def __getitem__(self, key: str):
        """支持字典式访问"""
        return self._get_nested(key)

    def __contains__(self, key: str) -> bool:
        """检查配置键是否存在"""
        try:
            self._get_nested(key)
            return True
        except (KeyError, TypeError):
            return False

    def reload(self) -> None:
        """重新加载配置文件"""
        self.config = self._load_config()

    def to_dict(self) -> dict[str]:
        """返回所有配置的字典形式"""
        return self.config.copy()
