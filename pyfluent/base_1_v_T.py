"""
PyFluent 自动化脚本：二维圆柱绕流 + 恒温加热

功能：
1. 批量扫描入口速度 / 入口温度
2. 自动计算
   - 最大速度
   - 最大温度
   - 入口平均压力
   - 出口平均压力
   - 压差 ΔP
3. 导出全局场 CSV
4. 导出 summary.csv

改进：不使用 report definitions，改用导出临时 CSV + Python 计算
"""

import ansys.fluent.core as pyfluent
import os
import csv
import pandas as pd

# ======================== 配置参数 ========================

MESH_FILE = r"D:\VScode\project\2026\demo1\2026-4-24\projectwbpj_files\dp0\FFF\Fluent\FFF.1-Setup-Output.cas.h5"
OUTPUT_DIR = r"D:\VScode\project\2026\demo1\2026-4-24\py\data"

ITERATIONS = 40

# 参数扫描（改成多个参数）
VELOCITIES   = [0.001,0.002,0.005,0.01,0.02,0.05,0.1]   # 1 mm/s
TEMPERATURES = [293.15, 303.15, 313.15, 323.15, 353.15] # 20 30 40 50 80℃

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ======================== 工具函数 ========================

def read_fluent_csv_robust(csv_path):
    """
    健壮地读取 Fluent 导出的 CSV
    尝试多种方式跳过注释行
    """
    # 尝试 1：不跳过（你的 CSV 第一行就是表头）
    try:
        df = pd.read_csv(csv_path, sep='\t')  # Fluent 用制表符分隔
        if not df.empty and len(df.columns) > 1:
            return df
    except:
        pass
    
    # 尝试 2：逗号分隔
    try:
        df = pd.read_csv(csv_path)
        if not df.empty and len(df.columns) > 1:
            return df
    except:
        pass
    
    # 尝试 3：跳过第一行
    try:
        df = pd.read_csv(csv_path, sep='\t', skiprows=1)
        if not df.empty and len(df.columns) > 1:
            return df
    except:
        pass
    
    return None

def compute_volume_stats(csv_path):
    """
    从流体域 CSV 计算 vmax, tmax
    """
    df = read_fluent_csv_robust(csv_path)
    
    if df is None:
        return {'vmax': -999.0, 'tmax': -999.0}
    
    # 清理列名：去除前后空格
    df.columns = df.columns.str.strip()
    
    result = {}
    
    # 查找速度列
    vel_col = None
    for col in df.columns:
        if 'velocity-magnitude' in col.lower():
            vel_col = col
            break
    
    if vel_col:
        result['vmax'] = float(df[vel_col].max())
    else:
        print(f"  ⚠️ 未找到速度列，可用列: {list(df.columns)}")
        result['vmax'] = -999.0
    
    # 查找温度列
    temp_col = None
    for col in df.columns:
        if 'temperature' in col.lower():
            temp_col = col
            break
    
    if temp_col:
        result['tmax'] = float(df[temp_col].max())
    else:
        print(f"  ⚠️ 未找到温度列，可用列: {list(df.columns)}")
        result['tmax'] = -999.0
    
    return result

def compute_surface_pressure(csv_path):
    """
    从表面 CSV 计算压力平均值
    """
    df = read_fluent_csv_robust(csv_path)
    
    if df is None:
        return -999.0
    
    # 清理列名：去除前后空格
    df.columns = df.columns.str.strip()
    
    # 查找压力列
    press_col = None
    for col in df.columns:
        if 'pressure' in col.lower():
            press_col = col
            break
    
    if press_col:
        return float(df[press_col].mean())
    else:
        print(f"  ⚠️ 未找到压力列，可用列: {list(df.columns)}")
        return -999.0

# ======================== 1. 启动 Fluent ========================

print("=" * 60)
print("启动 Fluent (2D, Double Precision)...")

solver = pyfluent.launch_fluent(
    precision="double",
    processor_count=4,
    mode="solver",
    show_gui=False,
    dimension=2
)

# ======================== 2. 读取 Case ========================

print("\n读取 Case 文件...")

solver.file.read_case(file_name=MESH_FILE)

# ======================== 3. 网格缩放 ========================

print("\n执行网格缩放: mm -> m")

solver.mesh.scale(
    x_scale=0.001,
    y_scale=0.001
)

# ======================== 4. 物理模型 ========================

print("\n设置物理模型与材料...")

solver.setup.models.viscous.model = "laminar"
solver.setup.models.energy.enabled = True

# 拷贝液态水材料
solver.tui.define.materials.copy(
    "fluid",
    "water-liquid"
)

# 指定流体区域材料
solver.setup.cell_zone_conditions.fluid["fff___"].material = "water-liquid"

# ======================== 5. 边界条件 ========================

print("\n设置边界条件...")

bc = solver.settings.setup.boundary_conditions

# outlet
bc.pressure_outlet["outlet"].momentum.gauge_pressure.value = 0

# 恒温壁
bc.wall["ball"].thermal.thermal_condition = "Temperature"
bc.wall["ball"].thermal.temperature.value = 368.15

bc.wall["heatwall"].thermal.thermal_condition = "Temperature"
bc.wall["heatwall"].thermal.temperature.value = 368.15

# 绝热壁
bc.wall["wall"].thermal.thermal_condition = "Heat Flux"
bc.wall["wall"].thermal.heat_flux.value = 0

# ======================== 6. 初始化 summary.csv ========================

summary_csv = os.path.join(OUTPUT_DIR, "summary.csv")

with open(summary_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow([
        "case_name",      # 新增：工况名称
        "v_inlet",
        "T_inlet",
        "vmax",
        "tmax",
        "p_in",
        "p_out",
        "delta_p"
    ])

print(f"已创建 summary.csv，表头：case_name, v_inlet, T_inlet, vmax, tmax, p_in, p_out, delta_p")

# ======================== 7. 工况循环 ========================

total = len(VELOCITIES) * len(TEMPERATURES)
count = 0

for v in VELOCITIES:
    for T in TEMPERATURES:
        count += 1

        print("\n" + "=" * 60)
        print(
            f"[{count}/{total}] "
            f"v_inlet = {v:.6f} m/s, "
            f"T_inlet = {T:.2f} K"
        )

        # ======================== 更新入口边界 ========================

        inlet = bc.velocity_inlet["inlet"]
        inlet.momentum.velocity.value = v
        inlet.thermal.temperature.value = T

        # ======================== 初始化 ========================

        print("初始化流场...")
        solver.settings.solution.initialization.hybrid_initialize()

        # ======================== 迭代 ========================

        print(f"开始迭代 {ITERATIONS} 步...")
        solver.settings.solution.run_calculation.iterate(iter_count=ITERATIONS)

        # ======================== 导出临时 CSV 计算统计量 ========================

        print("导出临时数据并计算统计量...")

        # 导出流体域数据（用于 vmax, tmax）
        temp_volume_csv = os.path.join(OUTPUT_DIR, "_temp_volume.csv")
        try:
            solver.settings.file.export.ascii(
                file_name=temp_volume_csv,
                surface_name_list=[],  # 空列表表示全域
                delimiter="comma",
                cell_func_domain=[
                    "velocity-magnitude",
                    "temperature",
                ],
                location="cell-center"
            )
            volume_stats = compute_volume_stats(temp_volume_csv)
            vmax = volume_stats['vmax']
            tmax = volume_stats['tmax']
            print(f"  流体域: vmax={vmax:.6e}, tmax={tmax:.6f}")
        except Exception as e:
            print(f"  流体域数据导出失败: {e}")
            vmax = -999.0
            tmax = -999.0

        # 导出入口表面数据（用于 p_in）
        temp_inlet_csv = os.path.join(OUTPUT_DIR, "_temp_inlet.csv")
        try:
            solver.settings.file.export.ascii(
                file_name=temp_inlet_csv,
                surface_name_list=["inlet"],
                delimiter="comma",
                cell_func_domain=["pressure"],  # 只导出压力
                location="cell-center"
            )
            p_in = compute_surface_pressure(temp_inlet_csv)
            print(f"  入口: p_in={p_in:.6e}")
        except Exception as e:
            print(f"  入口数据导出失败: {e}")
            p_in = -999.0

        # 导出出口表面数据（用于 p_out）
        temp_outlet_csv = os.path.join(OUTPUT_DIR, "_temp_outlet.csv")
        try:
            solver.settings.file.export.ascii(
                file_name=temp_outlet_csv,
                surface_name_list=["outlet"],
                delimiter="comma",
                cell_func_domain=["pressure"],  # 只导出压力
                location="cell-center"
            )
            p_out = compute_surface_pressure(temp_outlet_csv)
            print(f"  出口: p_out={p_out:.6e}")
        except Exception as e:
            print(f"  出口数据导出失败: {e}")
            p_out = -999.0

        # 压差
        delta_p = p_in - p_out

        # 清理临时文件
        for temp_file in [temp_volume_csv, temp_inlet_csv, temp_outlet_csv]:
            if os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except:
                    pass

        # ======================== 输出控制台 ========================

        print(
            f"📊 结果汇总: "
            f"vmax={vmax:.6e}, "
            f"tmax={tmax:.6f}, "
            f"p_in={p_in:.6e}, "
            f"p_out={p_out:.6e}, "
            f"ΔP={delta_p:.6e}"
        )

        # ======================== 写 summary.csv ========================

        # 生成工况名称（和全局场文件名保持一致）
        case_name = f"v{v:.4f}_T{T:.1f}"

        with open(summary_csv, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                case_name,    # 新增：工况名称
                v,
                T,
                vmax,
                tmax,
                p_in,
                p_out,
                delta_p
            ])

        # ======================== 导出全局场 CSV ========================

        fname = f"{case_name}.csv"  # 也可以用 case_name 作为文件名
        fpath = os.path.join(OUTPUT_DIR, fname)

        try:
            solver.settings.file.export.ascii(
                file_name=fpath,
                surface_name_list=[],
                delimiter="comma",
                cell_func_domain=[
                    "x-coordinate",
                    "y-coordinate",
                    "velocity-magnitude",
                    "temperature",
                    "pressure",
                ],
                location="cell-center"
            )

            print(f"✅ 全局场导出成功: {fname}")

        except Exception as e:
            print(f"❌ 全局场导出失败 ({fname})")
            print(e)

# ======================== 8. 退出 Fluent ========================

print("\n退出 Fluent...")
solver.exit()

print("\n" + "=" * 60)
print(f"全部 {total} 组工况已完成！")
print(f"数据目录：{OUTPUT_DIR}")
print(f"汇总文件：{summary_csv}")
