# MCP · 自然语言驱动的仿真优化Agent

在原有CFD代理模型与多目标优化的基础上，通过MCP协议将优化器和预测模型封装为标准化AI工具，实现自然语言直接驱动参数化设计寻优。

## 做了什么

```
自然语言 → Claude / Claude Code
               ↓ MCP协议
          MCP_server.py
               ↓
    ┌──────────────────────┐
    │  run_optimization    │ → NSGA-II多目标优化 → Pareto前沿
    │  predict_field       │ → RF代理模型 → 速度/温度/压力场预测
    │  get_best_design     │ → 按目标筛选 → 单个推荐设计点
    └──────────────────────┘
```

## 文件说明

| 文件 | 作用 |
|------|------|
| `MCP_optimization_nsga2.py` | 可导入版NSGA-II，原脚本封装成函数库 |
| `MCP_server.py` | MCP Server主文件，注册三个工具 |
| `.mcp.json` | Claude Code配置文件 |

## 使用方式

配置`.mcp.json`后重启Claude Code，直接对话：

```
帮我优化参数，速度范围0.001到0.1 m/s，温度范围293.15到353.15 K
预测x=0.005, y=0.003位置，入口速度0.05 m/s，温度313.15 K时的流场
```

## 技术栈

Python · MCP Protocol · pymoo(NSGA-II) · scikit-learn · LightGBM · joblib

## json设置
当前 MCP 服务配置共 3 个文件：

  1. 核心定义 — .mcp.json（项目根目录，不能改名）
  {
    "mcpServers": {
      "mcp-server": {
        "command": "D:\\...\\venv312\\Scripts\\python.exe",
        "args": ["...\\MCP_server.py"]
      }
    }
  }
  这是 MCP 服务的入口配置，指定了用哪个 Python 解释器跑哪个脚本。

  2. 启用开关 — .claude/settings.local.json
  { "enabledMcpjsonServers": ["mcp-server"] }
  告诉 Claude Code 启用 .mcp.json 里定义的 mcp-server，默认不写这个的话服务不会加载。

  3. C盘全局设置 — C:\Users\user\.claude\settings.json
  {
    "env": {
      "ANTHROPIC_BASE_URL": "https://api",
      "ANTHROPIC_AUTH_TOKEN": "sk-...",
      "ANTHROPIC_MODEL": ""
    },
    "theme": "dark",
    "includeCoAuthoredBy": false
  }
  这个是全局环境配置，定义了 API 地址和模型选择，与 MCP 无关。

  总结：要正常调用这个 MCP 服务，需要以上 3 个配置同时存在。.mcp.json 定义服务怎么跑，.claude/settings.local.json
  开启该服务，C:\Users\user\.claude\settings.json 提供全局 API 连接信息。
  

<img width="387.5" height="391" alt="image" src="https://github.com/user-attachments/assets/c101e01d-be50-44ab-a92e-9ad437162a06" />

