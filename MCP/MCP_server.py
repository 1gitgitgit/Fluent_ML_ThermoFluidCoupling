"""
mcp_server.py
CFD工业Agent · MCP Server

工具列表：
  run_optimization  : NSGA-II多目标寻优，返回完整Pareto前沿
  predict_field     : RF代理模型预测任意点的物理场
  get_best_design   : 优化后按指定目标返回单个推荐设计点

运行方式：
  python mcp_server.py           （直接运行，等待MCP协议输入）
  mcp dev mcp_server.py          （调试模式，自动打开Inspector）
"""

import sys
import os
import asyncio
import json
import traceback
import concurrent.futures

# ── 把项目根目录加入Python模块搜索路径 ──────────────────────────────────
PROJECT_ROOT = r"D:\VScode\project\2026\demo1"
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import joblib
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp import types

# ── 路径配置（和RF_1_baseline.py保持一致）────────────────────────────────
DATA_DIR = r"D:\VScode\project\2026\demo1\2026-4-24\py\data"

# ── 启动时加载全场代理模型（只加载一次）──────────────────────────────────
def _load_field_models():
    """
    加载RF_1_baseline.py保存的全场代理模型。
    注意：这里只用RF模型，因为MLP的StandardScaler未被保存，
    无法在新进程中还原。LGB模型可选用，精度与RF相近。
    """
    model_map = {
        "velocity":    "rf_vel.pkl",
        "temperature": "rf_temp.pkl",
        "pressure":    "rf_pres.pkl",
    }
    loaded = {}
    for field, fname in model_map.items():
        fpath = os.path.join(DATA_DIR, fname)
        if os.path.exists(fpath):
            loaded[field] = joblib.load(fpath)
            print(f"[Server] 加载模型: {fname}", file=sys.stderr)
        else:
            loaded[field] = None
            print(f"[Server] 警告：模型文件不存在: {fpath}", file=sys.stderr)
    return loaded

FIELD_MODELS = _load_field_models()

# ── 初始化MCP Server ──────────────────────────────────────────────────────
app = Server("cfd-industrial-agent")


# ════════════════════════════════════════════════════════════════════════════
# 注册工具列表
# ════════════════════════════════════════════════════════════════════════════

@app.list_tools()
async def list_tools() -> list[types.Tool]:
    return [

        # ── Tool 1 ──────────────────────────────────────────────────────
        types.Tool(
            name="run_optimization",
            description=(
                "运行NSGA-II多目标优化，在给定速度和温度范围内搜索Pareto最优解集。"
                "以随机森林代理模型为目标函数，同时最小化最大速度(vmax)和最大温度(tmax)。"
                "返回完整Pareto前沿（通常50-100个解），按tmax升序排列。"
                "适用场景：参数范围探索、多目标权衡分析、寻找散热最优参数等。"
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "v_min": {"type": "number", "description": "入口速度下限 (m/s)，如 0.001"},
                    "v_max": {"type": "number", "description": "入口速度上限 (m/s)，如 0.1"},
                    "T_min": {"type": "number", "description": "入口温度下限 (K)，如 293.15"},
                    "T_max": {"type": "number", "description": "入口温度上限 (K)，如 353.15"},
                    "n_gen": {
                        "type": "integer",
                        "description": "NSGA-II迭代代数，默认200，调试时可设50",
                        "default": 200
                    },
                },
                "required": ["v_min", "v_max", "T_min", "T_max"],
            },
        ),

        # ── Tool 2 ──────────────────────────────────────────────────────
        types.Tool(
            name="predict_field",
            description=(
                "用已训练的随机森林代理模型预测流场中任意点的物理量。"
                "可预测：速度场(velocity)、温度场(temperature)、压力场(pressure)。"
                "模型训练数据：35组CFD工况的全场数据，R²>0.95。"
                "适用场景：给定入口条件，快速预测场内任意位置的物理量。"
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "x":       {"type": "number", "description": "预测点x坐标 (m)"},
                    "y":       {"type": "number", "description": "预测点y坐标 (m)"},
                    "v_inlet": {"type": "number", "description": "入口速度 (m/s)"},
                    "T_inlet": {"type": "number", "description": "入口温度 (K)"},
                },
                "required": ["x", "y", "v_inlet", "T_inlet"],
            },
        ),

        # ── Tool 3 ──────────────────────────────────────────────────────
        types.Tool(
            name="get_best_design",
            description=(
                "运行NSGA-II优化后，按指定目标返回单个最优设计点。"
                "objective可选: 'min_tmax'(最低温度), 'min_vmax'(最低速度), 'balanced'(折中方案)。"
                "适用场景：用户需要明确一个推荐参数组合而非完整Pareto集时。"
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "objective": {
                        "type": "string",
                        "enum": ["min_tmax", "min_vmax", "balanced"],
                        "description": "优化目标：min_tmax最低温度 / min_vmax最低速度 / balanced折中",
                    },
                    "v_min": {"type": "number", "description": "入口速度下限 (m/s)"},
                    "v_max": {"type": "number", "description": "入口速度上限 (m/s)"},
                    "T_min": {"type": "number", "description": "入口温度下限 (K)"},
                    "T_max": {"type": "number", "description": "入口温度上限 (K)"},
                },
                "required": ["objective", "v_min", "v_max", "T_min", "T_max"],
            },
        ),
    ]


# ════════════════════════════════════════════════════════════════════════════
# 工具执行逻辑
# ════════════════════════════════════════════════════════════════════════════

@app.call_tool()
async def call_tool(name: str, arguments: dict):

    # ── Tool 1: run_optimization ─────────────────────────────────────────
    if name == "run_optimization":
        v_min = float(arguments["v_min"])
        v_max = float(arguments["v_max"])
        T_min = float(arguments["T_min"])
        T_max = float(arguments["T_max"])
        n_gen = int(arguments.get("n_gen", 200))

        # 参数合法性检查
        if v_min >= v_max:
            return [types.TextContent(type="text",
                text="参数错误：v_min 必须小于 v_max")]
        if T_min >= T_max:
            return [types.TextContent(type="text",
                text="参数错误：T_min 必须小于 T_max")]

        try:
            # 在线程池中运行同步的NSGA-II，避免阻塞事件循环
            loop = asyncio.get_event_loop()
            with concurrent.futures.ThreadPoolExecutor() as pool:
                results = await loop.run_in_executor(
                    pool,
                    lambda: _call_nsga2(v_min, v_max, T_min, T_max, n_gen)
                )

            return [types.TextContent(type="text", text=results)]

        except Exception as e:
            tb = traceback.format_exc()
            return [types.TextContent(type="text",
                text=f"优化运行出错：{e}\n\n详细信息：\n{tb}")]

    # ── Tool 2: predict_field ─────────────────────────────────────────────
    elif name == "predict_field":
        try:
            x       = float(arguments["x"])
            y       = float(arguments["y"])
            v_inlet = float(arguments["v_inlet"])
            T_inlet = float(arguments["T_inlet"])

            feature = np.array([[x, y, v_inlet, T_inlet]])

            lines = [
                f"物理场预测结果",
                f"  位置：x={x} m, y={y} m",
                f"  入口条件：v={v_inlet} m/s, T={T_inlet} K ({T_inlet-273.15:.2f}℃)",
                "─" * 45,
            ]

            field_labels = {
                "velocity":    ("速度场", "m/s"),
                "temperature": ("温度场", "K"),
                "pressure":    ("压力场", "Pa"),
            }

            for field, (cn_name, unit) in field_labels.items():
                model = FIELD_MODELS.get(field)
                if model is None:
                    lines.append(f"  {cn_name}：模型未加载（检查pkl文件路径）")
                else:
                    val = float(model.predict(feature).flatten()[0])
                    if field == "temperature":
                        lines.append(f"  {cn_name}：{val:.4f} {unit}  ({val-273.15:.2f}℃)")
                    else:
                        lines.append(f"  {cn_name}：{val:.6e} {unit}")

            return [types.TextContent(type="text", text="\n".join(lines))]

        except Exception as e:
            tb = traceback.format_exc()
            return [types.TextContent(type="text",
                text=f"预测出错：{e}\n\n详细信息：\n{tb}")]

    # ── Tool 3: get_best_design ───────────────────────────────────────────
    elif name == "get_best_design":
        objective = arguments["objective"]
        v_min = float(arguments["v_min"])
        v_max = float(arguments["v_max"])
        T_min = float(arguments["T_min"])
        T_max = float(arguments["T_max"])

        try:
            loop = asyncio.get_event_loop()
            with concurrent.futures.ThreadPoolExecutor() as pool:
                raw_results = await loop.run_in_executor(
                    pool,
                    lambda: _call_nsga2_raw(v_min, v_max, T_min, T_max, n_gen=200)
                )

            from MCP_optimization_nsga2 import get_recommended_designs
            recs = get_recommended_designs(raw_results)

            if objective not in recs:
                return [types.TextContent(type="text",
                    text=f"未知目标：{objective}，可选：min_tmax / min_vmax / balanced")]

            sol = recs[objective]
            objective_cn = {
                "min_tmax": "最低最大温度",
                "min_vmax": "最低最大速度",
                "balanced": "折中方案",
            }[objective]

            lines = [
                f"推荐设计点【{objective_cn}】",
                "─" * 45,
                f"  入口速度   v = {sol['v_inlet']:.5f} m/s",
                f"  入口温度   T = {sol['T_inlet']:.2f} K  ({sol['T_inlet_C']:.2f}℃)",
                f"  预测最大速度 vmax = {sol['vmax_pred']:.4f} m/s",
                f"  预测最大温度 tmax = {sol['tmax_pred']:.2f} K  ({sol['tmax_pred_C']:.2f}℃)",
            ]
            if "delta_p_pred" in sol:
                lines.append(f"  预测压差     ΔP   = {sol['delta_p_pred']:.2f} Pa")

            return [types.TextContent(type="text", text="\n".join(lines))]

        except Exception as e:
            tb = traceback.format_exc()
            return [types.TextContent(type="text",
                text=f"get_best_design出错：{e}\n\n{tb}")]

    else:
        return [types.TextContent(type="text",
            text=f"未知工具：{name}，可用工具：run_optimization / predict_field / get_best_design")]


# ════════════════════════════════════════════════════════════════════════════
# 内部辅助函数（同步，在ThreadPoolExecutor里调用）
# ════════════════════════════════════════════════════════════════════════════

def _call_nsga2(v_min, v_max, T_min, T_max, n_gen) -> str:
    """运行NSGA-II并返回格式化字符串（给run_optimization用）"""
    from MCP_optimization_nsga2 import run_nsga2
    results = run_nsga2(v_min, v_max, T_min, T_max, n_gen=n_gen, pop_size=100)

    lines = [
        f"NSGA-II优化完成，Pareto前沿共 {len(results)} 个解",
        f"搜索范围：v∈[{v_min},{v_max}] m/s | T∈[{T_min},{T_max}] K",
        "─" * 55,
        f"{'#':>3} {'v_inlet':>10} {'T(℃)':>8} {'vmax':>10} {'tmax(℃)':>10}",
        "─" * 55,
    ]

    # 展示前15个解（避免返回内容过长）
    display = results[:15]
    for i, r in enumerate(display):
        dp_str = f"  ΔP={r['delta_p_pred']:.1f}Pa" if "delta_p_pred" in r else ""
        lines.append(
            f"{i+1:>3} {r['v_inlet']:>10.5f} {r['T_inlet_C']:>8.2f} "
            f"{r['vmax_pred']:>10.4f} {r['tmax_pred_C']:>10.2f}{dp_str}"
        )

    if len(results) > 15:
        lines.append(f"  ... 还有 {len(results)-15} 个解（使用get_best_design获取推荐点）")

    # 附上三个推荐点摘要
    from MCP_optimization_nsga2 import get_recommended_designs
    recs = get_recommended_designs(results)
    lines += [
        "",
        "【推荐设计点摘要】",
        "─" * 55,
    ]
    for label, sol in recs.items():
        label_cn = {"min_tmax":"最低温度","min_vmax":"最低速度","balanced":"折中"}.get(label, label)
        lines.append(
            f"  [{label_cn}] v={sol['v_inlet']:.5f} m/s, "
            f"T={sol['T_inlet_C']:.2f}℃, "
            f"vmax={sol['vmax_pred']:.4f}, tmax={sol['tmax_pred_C']:.2f}℃"
        )

    return "\n".join(lines)


def _call_nsga2_raw(v_min, v_max, T_min, T_max, n_gen) -> list[dict]:
    """运行NSGA-II并返回原始结果（给get_best_design用）"""
    from MCP_optimization_nsga2 import run_nsga2
    return run_nsga2(v_min, v_max, T_min, T_max, n_gen=n_gen, pop_size=100)


# ════════════════════════════════════════════════════════════════════════════
# 启动入口
# ════════════════════════════════════════════════════════════════════════════

async def main():
    async with stdio_server() as (read, write):
        await app.run(read, write, app.create_initialization_options())

if __name__ == "__main__":
    asyncio.run(main())