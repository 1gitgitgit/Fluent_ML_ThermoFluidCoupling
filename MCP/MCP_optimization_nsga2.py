"""
optimization_nsga2_func.py
可导入版NSGA-II优化模块，供mcp_server.py调用。

改动说明（相对于optimization_nsga2.py）：
1. matplotlib.use('Agg') 阻止GUI弹窗卡死Server
2. 所有逻辑封装进 run_nsga2() 函数 ！！这是重点，做成了类似numpy的库
3. verbose=False 防止日志污染MCP的stdout通信
4. plt.show() 改为 plt.savefig() 或跳过
5. 返回结构化list[dict]而非打印
"""

import matplotlib
matplotlib.use('Agg')  # ← 必须在pyplot之前，阻止GUI弹窗

import pandas as pd
import numpy as np
import time
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, cross_val_score
import lightgbm as lgb
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.optimize import minimize
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling

import warnings
warnings.filterwarnings("ignore")

# ── 默认路径（和RF_1_baseline.py保持一致）────────────────────────────────
DEFAULT_DATA_DIR  = r"D:\VScode\project\2026\demo1\2026-4-24\py\data"
DEFAULT_SUMMARY   = rf"{DEFAULT_DATA_DIR}\summary.csv"


def run_nsga2(
    v_min: float,
    v_max: float,
    T_min: float,
    T_max: float,
    summary_csv: str = DEFAULT_SUMMARY,
    pop_size: int = 100,
    n_gen: int = 200,
    save_plots: bool = False,       # True时保存图片到data目录，False时跳过
    data_dir: str = DEFAULT_DATA_DIR,
) -> list[dict]:
    """
    运行NSGA-II多目标优化，返回Pareto前沿解集。

    Args:
        v_min, v_max : 入口速度搜索范围 (m/s)
        T_min, T_max : 入口温度搜索范围 (K)
        summary_csv  : summary.csv路径，含35组工况的vmax/tmax
        pop_size     : 种群大小（调试时可改50加速）
        n_gen        : 迭代代数（调试时可改50加速）
        save_plots   : 是否保存响应面和Pareto图到文件（不弹窗）
        data_dir     : 图片保存目录

    Returns:
        list[dict]，每个dict包含:
            v_inlet     : 入口速度 (m/s)
            T_inlet     : 入口温度 (K)
            T_inlet_C   : 入口温度 (℃)
            vmax_pred   : 预测最大速度 (m/s)
            tmax_pred   : 预测最大温度 (K)
            tmax_pred_C : 预测最大温度 (℃)
            delta_p_est : 压差估计（如summary.csv含delta_p列则填入，否则None）
        按 tmax_pred 升序排列（最低温度在前）
    """

    # ── 1. 读取summary数据 ────────────────────────────────────────────────
    df = pd.read_csv(summary_csv)

    # 检查必要列
    required_cols = {"v_inlet", "T_inlet", "vmax", "tmax"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"summary.csv缺少必要列: {missing}，实际列: {list(df.columns)}")

    X      = df[["v_inlet", "T_inlet"]].values
    y_vmax = df["vmax"].values
    y_tmax = df["tmax"].values

    # 如果有delta_p列，也训练一个代理（用于结果enrichment）
    has_delta_p = "delta_p" in df.columns
    if has_delta_p:
        y_dp = df["delta_p"].values

    # ── 2. 训练工况级代理模型 ─────────────────────────────────────────────
    def _train_and_pick(X, y, label=""):
        """训练RF和LGB，返回全量R²更高的那个"""
        rf = RandomForestRegressor(
            n_estimators=300, max_depth=10, n_jobs=-1, random_state=42
        )
        lgb_m = lgb.LGBMRegressor(
            n_estimators=300, learning_rate=0.05,
            num_leaves=31, random_state=42, verbose=-1
        )
        rf.fit(X, y)
        lgb_m.fit(X, y)
        r2_rf  = r2_score(y, rf.predict(X))
        r2_lgb = r2_score(y, lgb_m.predict(X))
        winner = rf if r2_rf >= r2_lgb else lgb_m
        return winner

    surrogate_vmax = _train_and_pick(X, y_vmax, "vmax")
    surrogate_tmax = _train_and_pick(X, y_tmax, "tmax")
    surrogate_dp   = _train_and_pick(X, y_dp, "delta_p") if has_delta_p else None

    # ── 3. 定义多目标优化问题 ─────────────────────────────────────────────
    class CoolingOptProblem(Problem):
        def __init__(self, model_vmax, model_tmax):
            super().__init__(
                n_var=2,
                n_obj=2,
                n_ieq_constr=0,
                xl=np.array([v_min, T_min]),
                xu=np.array([v_max, T_max]),
            )
            self.model_vmax = model_vmax
            self.model_tmax = model_tmax

        def _evaluate(self, x, out, *args, **kwargs):
            f1 = self.model_vmax.predict(x)
            f2 = self.model_tmax.predict(x)
            out["F"] = np.column_stack([f1, f2])

    problem = CoolingOptProblem(surrogate_vmax, surrogate_tmax)

    # ── 4. 运行NSGA-II ───────────────────────────────────────────────────
    algorithm = NSGA2(
        pop_size=pop_size,
        sampling=FloatRandomSampling(),
        crossover=SBX(prob=0.9, eta=15),
        mutation=PM(eta=20),
        eliminate_duplicates=True,
    )

    res = minimize(
        problem,
        algorithm,
        termination=("n_gen", n_gen),
        seed=42,
        verbose=False,   
        #  关键：False防止日志污染MCP stdout
    )

    pareto_X = res.X   # (n_pareto, 2): v_inlet, T_inlet
    pareto_F = res.F   # (n_pareto, 2): vmax_pred, tmax_pred

    # ── 5. 可选：保存图片（不弹窗）───────────────────────────────────────
    if save_plots:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(pareto_F[:, 0], pareto_F[:, 1], c="steelblue", s=15)
        ax.set_xlabel("vmax_pred (m/s)")
        ax.set_ylabel("tmax_pred (K)")
        ax.set_title(f"Pareto Front | v∈[{v_min},{v_max}] T∈[{T_min},{T_max}]")
        plt.tight_layout()
        plt.savefig(rf"{data_dir}\pareto_mcp_run.png", dpi=120)
        plt.close()

    # ── 6. 整理结果为list[dict] ──────────────────────────────────────────
    results = []
    for i in range(len(pareto_X)):
        v   = float(pareto_X[i, 0])
        T   = float(pareto_X[i, 1])
        vm  = float(pareto_F[i, 0])
        tm  = float(pareto_F[i, 1])

        entry = {
            "v_inlet":     round(v,  6),
            "T_inlet":     round(T,  4),
            "T_inlet_C":   round(T - 273.15, 2),
            "vmax_pred":   round(vm, 6),
            "tmax_pred":   round(tm, 4),
            "tmax_pred_C": round(tm - 273.15, 2),
        }

        # 如果有delta_p代理，额外预测压差
        if surrogate_dp is not None:
            dp = float(surrogate_dp.predict([[v, T]])[0])
            entry["delta_p_pred"] = round(dp, 4)

        results.append(entry)

    # 按 tmax 升序排列
    results.sort(key=lambda r: r["tmax_pred"])
    return results


def get_recommended_designs(pareto_results: list[dict]) -> dict:
    """
    从Pareto结果中提取三个推荐设计点。

    Args:
        pareto_results: run_nsga2()的返回值

    Returns:
        dict，包含三个key:
            min_tmax    : 最低最大温度方案
            min_vmax    : 最低最大速度方案
            balanced    : 归一化距离原点最近的折中方案
    """
    if not pareto_results:
        return {}

    vmax_vals = np.array([r["vmax_pred"] for r in pareto_results])
    tmax_vals = np.array([r["tmax_pred"] for r in pareto_results])

    idx_min_tmax = int(np.argmin(tmax_vals))
    idx_min_vmax = int(np.argmin(vmax_vals))

    F = np.column_stack([vmax_vals, tmax_vals])
    F_norm = (F - F.min(axis=0)) / (F.max(axis=0) - F.min(axis=0) + 1e-12)
    idx_balanced = int(np.argmin(np.linalg.norm(F_norm, axis=1)))

    return {
        "min_tmax": pareto_results[idx_min_tmax],
        "min_vmax": pareto_results[idx_min_vmax],
        "balanced": pareto_results[idx_balanced],
    }


# ── 本地测试入口（直接运行此文件时） ─────────────────────────────────────
if __name__ == "__main__":
    print("测试 run_nsga2()...")
    results = run_nsga2(
        v_min=0.001, v_max=0.1,
        T_min=293.15, T_max=353.15,
        pop_size=50,    # 调试用小参数
        n_gen=50,
    )
    print(f"Pareto前沿共 {len(results)} 个解")
    print("前5个解（按tmax排序）:")
    for i, r in enumerate(results[:5]):
        print(f"  #{i+1}: v={r['v_inlet']:.5f} m/s, T={r['T_inlet_C']:.1f}℃, "
              f"vmax={r['vmax_pred']:.4f}, tmax={r['tmax_pred_C']:.1f}℃")

    recs = get_recommended_designs(results)
    print("\n推荐设计点:")
    for label, sol in recs.items():
        print(f"  [{label}] v={sol['v_inlet']:.5f} m/s, T={sol['T_inlet_C']:.1f}℃")