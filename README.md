# Fluent-ML Thermo-Fluid Coupling Optimization

基于 ANSYS Fluent 的热流耦合仿真数据，构建机器学习代理模型，并结合 NSGA-II 多目标优化搜索入口速度与入口温度的 Pareto 最优设计方案。

这个项目的核心目标不是替代 CFD 的物理求解过程，而是把高成本的 Fluent 参数扫描结果沉淀为可快速查询的代理模型，用于工程参数探索、趋势分析和优化决策。

## 项目背景

热流耦合问题中，入口速度、入口温度等边界条件会同时影响流场、温度场和压力损失。直接依赖 Fluent 进行参数搜索时，每一个设计点都需要重新求解，成本较高，不适合快速比较大量候选方案。

本项目围绕一个二维圆柱绕流与恒温加热场景，完成了以下闭环：

1. 使用 PyFluent 自动化批量仿真，生成不同入口条件下的 CFD 数据。
2. 从 Fluent 导出的全场 CSV 中提取速度、温度、压力等物理量。
3. 训练 Random Forest、LightGBM、MLP、GPR 等代理模型，预测任意空间点的物理场。
4. 基于工况级指标训练 surrogate model，使用 NSGA-II 搜索 Pareto 最优入口参数。
5. 将预测与优化能力封装为 MCP Server，支持 Agent 工具化调用。

## 技术栈

- CFD 仿真：ANSYS Fluent, PyFluent
- 数据处理：Python, pandas, NumPy
- 机器学习：scikit-learn, LightGBM, PyTorch/MLP, Gaussian Process Regression
- 优化算法：pymoo, NSGA-II
- 可视化：Matplotlib
- Agent 工具化：MCP Server

## 数据与任务定义

参数扫描范围：

| 变量 | 范围 |
| --- | --- |
| 入口速度 `v_inlet` | 0.001 - 0.1 m/s |
| 入口温度 `T_inlet` | 293.15 - 353.15 K |

数据规模：

| 项目 | 数量 |
| --- | ---: |
| CFD 工况数 | 35 |
| 全场样本点 | 147,700 |
| 输入特征 | `x`, `y`, `v_inlet`, `T_inlet` |
| 预测目标 | `velocity`, `temperature`, `pressure` |

工况级优化指标：

- `vmax`：流场最大速度
- `tmax`：流场最高温度
- `delta_p`：入口与出口压差

## 方法流程

```text
PyFluent 批量仿真
        |
        v
Fluent CSV 场数据导出
        |
        v
数据清洗与特征构造
        |
        v
训练速度 / 温度 / 压力代理模型
        |
        v
模型评估与边界工况验证
        |
        v
NSGA-II 多目标优化
        |
        v
Pareto 前沿与推荐设计点
        |
        v
MCP Server 工具化封装
```

## 模型结果

随机划分测试集上的全场预测结果如下：

| Model | R2 | RMSE | NRMSE |
| --- | ---: | ---: | ---: |
| RF - Velocity | 0.9996 | 7.7820e-04 | 0.3293% |
| RF - Temperature | 0.9999 | 1.6963e-01 | 0.2262% |
| RF - Pressure | 1.0000 | 4.4561e-01 | 0.0164% |
| LGB - Velocity | 0.9999 | 4.2870e-04 | 0.1814% |
| LGB - Temperature | 0.9999 | 1.8931e-01 | 0.2524% |
| LGB - Pressure | 1.0000 | 1.0642e+00 | 0.0392% |
| GPR - Velocity | 0.9335 | 1.0059e-02 | 4.2567% |
| GPR - Temperature | 0.9916 | 2.0561e+00 | 2.7415% |
| GPR - Pressure | 0.9879 | 5.6183e+01 | 2.0704% |

在随机划分下，Random Forest 与 LightGBM 对场变量具有较高拟合精度，可以作为快速代理模型使用。进一步的边界工况外推实验显示，模型在未见过的边界组合上性能会下降，这说明代理模型在工程应用中需要关注训练样本覆盖范围，而不能只依赖随机划分指标。

## 多目标优化

优化变量：

- `v_inlet`
- `T_inlet`

优化目标：

- 最小化 `vmax`
- 最小化 `tmax`

项目使用 NSGA-II 搜索入口速度和入口温度的 Pareto 前沿，用于分析不同散热目标之间的权衡关系。优化结果会生成：

- `response_surface.png`：代理模型响应面
- `pareto_front.png`：Pareto 前沿
- `pareto_results_valid.csv`：有效优化结果
- `feasibility_region_tmax320.png`：约束可行域分析图

## MCP 工具化封装

项目额外实现了一个 MCP Server，将 CFD 代理模型和优化流程封装为 Agent 可调用工具：

| Tool | 功能 |
| --- | --- |
| `predict_field` | 给定空间坐标和入口条件，预测速度、温度、压力 |
| `run_optimization` | 在给定参数范围内运行 NSGA-II，返回 Pareto 解集 |
| `get_best_design` | 按最低温度、最低速度或折中目标返回推荐设计点 |

这部分体现了从离线仿真数据到智能体工具调用的工程闭环：仿真结果不只停留在图表和模型文件中，而是被封装成可交互的工程能力。

## 项目亮点

- 使用 PyFluent 自动化生成多工况热流耦合仿真数据，减少手工操作成本。
- 构建速度场、温度场、压力场代理模型，实现 CFD 场变量的快速预测。
- 对比 Random Forest、LightGBM、MLP、GPR 等模型，并记录精度与泛化差异。
- 引入 Reynolds 数等物理相关特征，分析模型在边界工况外推时的表现。
- 使用 NSGA-II 将代理模型用于多目标参数优化，生成 Pareto 前沿。
- 通过 MCP Server 将预测和优化能力封装为 Agent 工具，形成“仿真 - 建模 - 优化 - 调用”的完整流程。

## 主要文件

| 文件 | 说明 |
| --- | --- |
| `base_1.py` | PyFluent 自动化仿真与数据导出 |
| `RF_1_baseline.py` | 基线代理模型训练与评估 |
| `RF_2_GridSearchCV.py` | 特征工程与模型调参实验 |
| `Torch_1_MLP.py` | MLP 模型实验 |
| `optimization_nsga2.py` | 工况级代理模型与 NSGA-II 多目标优化 |
| `mcp/MCP_server.py` | MCP Server 工具封装 |
| `data/summary.csv` | CFD 工况级统计结果 |
| `data/model_comparison.csv` | 代理模型对比结果 |
| `data/pareto_results_valid.csv` | Pareto 优化结果 |

## 可用于简历的表述

- 基于 PyFluent 搭建热流耦合 CFD 自动化仿真流程，批量生成 35 组入口速度/温度工况与约 14.7 万个全场样本点。
- 使用 Random Forest、LightGBM、MLP、GPR 构建速度场、温度场、压力场代理模型，随机测试集上主要模型 R2 达到 0.999 量级。
- 设计边界工况外推实验，分析代理模型在未见工况下的泛化能力，并引入 Reynolds 数等物理特征进行改进。
- 基于 surrogate model 和 NSGA-II 实现入口参数多目标优化，搜索最大速度与最高温度之间的 Pareto 最优解。
- 将预测与优化流程封装为 MCP Server，支持 Agent 通过 `predict_field`、`run_optimization`、`get_best_design` 等工具调用工程模型。

## 当前限制

- 当前代理模型主要依赖有限数量的 Fluent 工况，外推能力受训练样本覆盖范围影响。
- 大型模型文件和原始数据文件体积较大，不适合直接纳入普通 Git 仓库。
- 项目目前更偏研究验证与作品集展示，若用于生产级工程优化，还需要更系统的误差估计、不确定性分析和更多仿真样本验证。



<img width="563" height="150" alt="feature_importance" src="https://github.com/user-attachments/assets/a2e14f1d-23f9-4879-ad2d-5fb040ea6914" />
<img width="562.5" height="487.5" alt="pred_vs_true" src="https://github.com/user-attachments/assets/eccc8633-a9d2-4fab-89c4-ea9f496675da" />
<img width="563" height="375" alt="field_comparison" src="https://github.com/user-attachments/assets/e724c822-f6de-48f2-bb8d-d4c7abf34fb5" />



