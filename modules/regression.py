from matplotlib.axes import Axes
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyparsing import Dict
import shap
from sklearn.discriminant_analysis import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    r2_score,
    root_mean_squared_error,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from matplotlib import rcParams
import matplotlib
import seaborn as sns
import os
from scipy.optimize import curve_fit, fsolve

from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

from xgboost import XGBRegressor

from utils.filepath import get_temp_image_path


matplotlib.rcParams["font.sans-serif"] = ["SimHei"]
matplotlib.rcParams["axes.unicode_minus"] = False


class RegressionModel:
    def __init__(
        self,
        model: str,
        data: pd.DataFrame,
        feature: list[str],
        target: str,
        test_size: float = 0.2,
        random_state: int | None = None,
    ):
        """
        初始化回归模型。

        Args:
            model (str): 模型类型字符串
            data (pd.DataFrame): 数据集
            feature (list[str]): 特征列表
            target (str): 目标特征
            test_size (float, optional): 测试集占数据集比例. Defaults to 0.2.
            random_state (int | None, optional): 随机数种子. Defaults to None.
        """

        self.data = data
        self.feature = feature
        self.target = target
        self.X = data[feature]
        self.y = data[target]
        self.test_size = test_size
        self.random_state = random_state
        self.scaler = StandardScaler().set_output(transform="pandas")
        self.model = self.__load_model(model)
        self.is_trained = False
        self.shap_values = None

    def __load_model(self, model: str):
        if model == "Ridge":
            return Ridge()
        elif model == "Lasso":
            return Lasso()
        elif model == "SVR":
            return SVR()
        elif model == "DecisionTree":
            return DecisionTreeRegressor()
        elif model == "RandomForest":
            return RandomForestRegressor()
        elif model == "XGBoost":
            return XGBRegressor()
        elif model == "LinearRegression":
            return LinearRegression()
        else:
            raise ValueError(f"不支持的模型类型: {model}")

    def train_and_evaluate_model(self):
        """
        训练回归模型并评估其性能。

        返回：
        - 包含 MAE & RMSE & R² 的字典。
        """
        self.shap_values = None

        # 将数据集划分为训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=self.test_size, random_state=self.random_state)

        # 训练模型
        X_train_scaled = self.scaler.fit_transform(X_train)
        self.model.fit(X_train_scaled, y_train)

        # 进行预测
        X_test_scaled = self.scaler.transform(X_test)
        y_test_pred = self.model.predict(X_test_scaled)
        y_train_pred = self.model.predict(X_train_scaled)

        # 计算评估指标
        rmse = root_mean_squared_error(y_test, y_test_pred)
        mae = mean_absolute_error(y_test, y_test_pred)
        r2 = r2_score(y_test, y_test_pred)

        # 存储结果
        self.results = {
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
            "y_train": y_train,
            "y_train_pred": y_train_pred,
            "y_test": y_test,
            "y_test_pred": y_test_pred,
        }

        self.is_trained = True

        return self.results

    def shap_importance(self):
        """
        计算SHAP值并返回特征重要性。

        返回：
        - 包含特征重要性的DataFrame
        """
        if not self.is_trained:
            raise ValueError("模型未训练，请先调用 train_and_evaluate_model 方法。")

        if not self.shap_values:
            X_scaled = self.scaler.transform(self.X)
            explainer = shap.Explainer(self.model.predict, X_scaled)
            self.shap_values = explainer(X_scaled)

        return self.shap_values

    def plot_shap_importance(self):
        """
        绘制基于 SHAP 值的特征重要性图。
        """
        if not self.shap_values:
            X_scaled = self.scaler.transform(self.X)
            explainer = shap.Explainer(self.model.predict, X_scaled)
            self.shap_values = explainer(X_scaled)

        fig, ax = plt.subplots(figsize=(12, 10))

        shap.summary_plot(self.shap_values, X_scaled, plot_type="bar", show=False)
        ax.set_title("基于 SHAP 的特征重要性分析")
        ax.set_xlabel("特征重要性")
        ax.set_ylabel("特征")

        img_path = os.path.join(get_temp_image_path(), "shap_importance.png")
        fig.savefig(img_path)

        plt.show()

        plt.close()
        return img_path


def find_best_model(
    data: pd.DataFrame,
    feature: list[str],
    target: str,
    test_size: float = 0.2,
    random_state: int | None = None,
):
    """
    找到最佳模型。

    返回：
    - 包含所有模型评估指标的 DataFrame
    - RMSE最小的模型名称
    """
    models_list = ["Ridge", "Lasso", "SVR", "DecisionTree", "RandomForest", "XGBoost", "LinearRegression"]
    results = []
    models: dict[str, RegressionModel] = {}

    # 遍历所有模型并评估性能
    for model_name in models_list:
        try:
            # 创建模型实例
            reg_model = RegressionModel(
                model=model_name, data=data, feature=feature, target=target, test_size=test_size, random_state=random_state
            )

            # 训练并评估模型
            eval_result = reg_model.train_and_evaluate_model()

            # 收集评估结果
            results.append(
                {
                    "Models": model_name,
                    "RMSE": round(eval_result["rmse"], 4),
                    "MAE": round(eval_result["mae"], 4),
                    "R2": round(eval_result["r2"], 4),
                }
            )
            models[model_name] = reg_model

        except Exception as e:
            print(f"模型 {model_name} 训练失败: {str(e)}")
            continue

    # 转换为DataFrame格式
    results_df = pd.DataFrame(results)

    # 确定RMSE最小的最佳模型
    if not results_df.empty:
        best_model = results_df.loc[results_df["RMSE"].idxmin(), "Models"]
    else:
        best_model = None
        print("所有模型训练失败，无法确定最佳模型")

    return results_df, best_model, models


def plot_multi_types(results: dict[str, dict]):
    """
    绘制多个类型的 RMSE 对比图。

    """
    fig, ax = plt.subplots(figsize=(12, 10))
    x = []
    y = []
    for type, result in results.items():
        x.append(type)
        y.append(result["rmse"])

    # 绘制柱状图
    ax.bar(x, y)

    ax.set_title("多个类型的 RMSE 对比图")
    ax.set_xlabel("类型")
    ax.set_ylabel("RMSE")
    ax.legend()
    img_path = os.path.join(get_temp_image_path(), "multi_types_rmse.png")
    fig.savefig(img_path)

    plt.show()

    plt.close()
    return img_path
