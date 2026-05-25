import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern
import lightgbm as lgb


# ========= 1. 配置 =========
DATA_DIR     = r"D:\VScode\project\2026\demo1\2026-4-24\py\data"
VELOCITIES   = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1]
TEMPERATURES = [293.15, 303.15, 313.15, 323.15, 353.15]
GPR_SUBSAMPLE = 2000   # GPR子采样数量，内存限制下2000点稳定可跑

# ========= 2. 读取全局场数据 =========
dfs = []
for v in VELOCITIES:
    for T in TEMPERATURES:
        fname = f"v{v:.4f}_T{T:.1f}.csv"
        fpath = f"{DATA_DIR}\\{fname}"
        try:
            df = pd.read_csv(fpath, sep=",", skiprows=1, header=None)
            df.columns = [
                "cellnumber", "x", "y", "x2", "y2",
                "vel_mag", "temperature", "pressure"
            ]
            original_count = len(df)
            df = df.replace([np.inf, -np.inf], np.nan).dropna()
            dropped   = original_count - len(df)
            drop_rate = dropped / original_count
            print(f"工况 v={v}, T={T}: {len(df)} 个点, "
                  f"清洗 {dropped} 个 ({drop_rate:.2%})")
            if drop_rate > 0.05:
                print(f"  ❗ 警告: 剔除比例过高 ({drop_rate:.2%})")
            df["v_inlet"] = v
            df["T_inlet"] = T
            df = df.drop(columns=["cellnumber", "x2", "y2"])
            dfs.append(df)
        except FileNotFoundError:
            print(f"  ⚠ 文件不存在，跳过: {fname}")

data = pd.concat(dfs, ignore_index=True)
print("-" * 40)
print(f"数据合并完成，总样本数: {len(data)}")

# ========= 3. 特征 & 标签 =========
FEATURES = ["x", "y", "v_inlet", "T_inlet"]
X      = data[FEATURES]
y_vel  = data["vel_mag"]
y_temp = data["temperature"]
y_pres = data["pressure"]

# ========= 4. 划分数据 =========
X_train, X_test, \
yv_train, yv_test, \
yt_train, yt_test, \
yp_train, yp_test = train_test_split(
    X, y_vel, y_temp, y_pres,
    test_size=0.2, random_state=42
)
print(f"训练集: {len(X_train)}  测试集: {len(X_test)}")

# ========= 5. 评估函数 =========
def evaluate(model, X_test, y_test, name, scaler_y=None):
    t0     = time.time()
    y_pred = model.predict(X_test)
    t1     = time.time()
    if scaler_y is not None:
        y_pred = scaler_y.inverse_transform(
                     y_pred.reshape(-1, 1)).ravel()
        y_test = scaler_y.inverse_transform(
                     np.array(y_test).reshape(-1, 1)).ravel()
    rmse  = np.sqrt(mean_squared_error(y_test, y_pred))
    r2    = r2_score(y_test, y_pred)
    nrmse = rmse / (y_test.max() - y_test.min())
    print(f"  [{name}]  耗时: {t1-t0:.4f}s | "
          f"RMSE: {rmse:.4e} | NRMSE: {nrmse:.4%} | R²: {r2:.6f}")
    return y_pred, np.array(y_test)

# ========= 6. Random Forest =========
print("\n" + "="*45)
print("===== Random Forest =====")
print("="*45)

rf_vel  = RandomForestRegressor(n_estimators=200, max_depth=15,
                                 n_jobs=-1, random_state=42)
rf_temp = RandomForestRegressor(n_estimators=200, max_depth=15,
                                 n_jobs=-1, random_state=42)
rf_pres = RandomForestRegressor(n_estimators=200, max_depth=15,
                                 n_jobs=-1, random_state=42)

total_t = time.time()
for model, ytr, name in [
    (rf_vel,  yv_train, "RF-Velocity"),
    (rf_temp, yt_train, "RF-Temperature"),
    (rf_pres, yp_train, "RF-Pressure"),
]:
    t0 = time.time()
    print(f"  训练 {name}...")
    model.fit(X_train, ytr)
    print(f"  训练耗时: {time.time()-t0:.2f}s")
print(f"RF 总训练耗时: {time.time()-total_t:.2f}s\n")

rv_pred, yv_true = evaluate(rf_vel,  X_test, yv_test, "RF Velocity (m/s)")
rt_pred, yt_true = evaluate(rf_temp, X_test, yt_test, "RF Temperature (K)")
rp_pred, yp_true = evaluate(rf_pres, X_test, yp_test, "RF Pressure (Pa)")

# ========= 7. LightGBM =========
print("\n" + "="*45)
print("===== LightGBM =====")
print("="*45)

lgb_vel  = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.05,
                               num_leaves=63, n_jobs=-1,
                               random_state=42, verbose=-1)
lgb_temp = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.05,
                               num_leaves=63, n_jobs=-1,
                               random_state=42, verbose=-1)
lgb_pres = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.05,
                               num_leaves=63, n_jobs=-1,
                               random_state=42, verbose=-1)

total_t = time.time()
for model, ytr, name in [
    (lgb_vel,  yv_train, "LGB-Velocity"),
    (lgb_temp, yt_train, "LGB-Temperature"),
    (lgb_pres, yp_train, "LGB-Pressure"),
]:
    t0 = time.time()
    print(f"  训练 {name}...")
    model.fit(X_train, ytr)
    print(f"  训练耗时: {time.time()-t0:.2f}s")
print(f"LightGBM 总训练耗时: {time.time()-total_t:.2f}s\n")

lv_pred, _ = evaluate(lgb_vel,  X_test, yv_test, "LGB Velocity (m/s)")
lt_pred, _ = evaluate(lgb_temp, X_test, yt_test, "LGB Temperature (K)")
lp_pred, _ = evaluate(lgb_pres, X_test, yp_test, "LGB Pressure (Pa)")

# ========= 8. Kriging (Gaussian Process Regression) =========
print("\n" + "="*45)
print("===== Kriging (GPR) =====")
print("="*45)
print(f"  GPR数据量大，对训练集子采样至 {GPR_SUBSAMPLE} 个点")

# 子采样（固定随机种子保证可复现）
rng = np.random.default_rng(42)
idx = rng.choice(len(X_train), size=min(GPR_SUBSAMPLE, len(X_train)),
                 replace=False)
X_train_gpr  = X_train.iloc[idx]
yv_train_gpr = yv_train.iloc[idx]
yt_train_gpr = yt_train.iloc[idx]
yp_train_gpr = yp_train.iloc[idx]

# GPR对量纲敏感，做标准化
scaler_X_gpr  = StandardScaler()
scaler_yv_gpr = StandardScaler()
scaler_yt_gpr = StandardScaler()
scaler_yp_gpr = StandardScaler()

X_train_gpr_s = scaler_X_gpr.fit_transform(X_train_gpr)
X_test_gpr_s  = scaler_X_gpr.transform(X_test)

yv_train_gpr_s = scaler_yv_gpr.fit_transform(
    yv_train_gpr.values.reshape(-1, 1)).ravel()
yt_train_gpr_s = scaler_yt_gpr.fit_transform(
    yt_train_gpr.values.reshape(-1, 1)).ravel()
yp_train_gpr_s = scaler_yp_gpr.fit_transform(
    yp_train_gpr.values.reshape(-1, 1)).ravel()

# Matern核：对物理场比RBF更鲁棒（nu=2.5对应二阶可微）
kernel = ConstantKernel(1.0, constant_value_bounds=(1e-3, 1e3)) * \
         Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e2), nu=2.5)

gpr_vel  = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5,
                                     normalize_y=False, random_state=42)
gpr_temp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5,
                                     normalize_y=False, random_state=42)
gpr_pres = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5,
                                     normalize_y=False, random_state=42)

total_t = time.time()
for model, ytr_s, name in [
    (gpr_vel,  yv_train_gpr_s, "GPR-Velocity"),
    (gpr_temp, yt_train_gpr_s, "GPR-Temperature"),
    (gpr_pres, yp_train_gpr_s, "GPR-Pressure"),
]:
    t0 = time.time()
    print(f"  训练 {name}...")
    model.fit(X_train_gpr_s, ytr_s)
    print(f"  训练耗时: {time.time()-t0:.2f}s")
print(f"GPR 总训练耗时: {time.time()-total_t:.2f}s\n")

# GPR预测（还原标准化 + 输出不确定度）
def evaluate_gpr(model, X_test_s, y_test, scaler_y, name):
    t0 = time.time()
    y_pred_s, sigma_s = model.predict(X_test_s, return_std=True)
    t1 = time.time()

    y_pred = scaler_y.inverse_transform(y_pred_s.reshape(-1, 1)).ravel()
    sigma  = sigma_s * scaler_y.scale_[0]   # 不确定度同步还原量纲
    y_test = np.array(y_test)

    rmse  = np.sqrt(mean_squared_error(y_test, y_pred))
    r2    = r2_score(y_test, y_pred)
    nrmse = rmse / (y_test.max() - y_test.min())
    mean_sigma = sigma.mean()

    print(f"  [{name}]  耗时: {t1-t0:.4f}s | "
          f"RMSE: {rmse:.4e} | NRMSE: {nrmse:.4%} | R²: {r2:.6f} | "
          f"平均不确定度σ: {mean_sigma:.4e}")
    return y_pred, sigma, y_test

gv_pred, gv_sigma, _ = evaluate_gpr(gpr_vel,  X_test_gpr_s, yv_test,
                                     scaler_yv_gpr, "GPR Velocity (m/s)")
gt_pred, gt_sigma, _ = evaluate_gpr(gpr_temp, X_test_gpr_s, yt_test,
                                     scaler_yt_gpr, "GPR Temperature (K)")
gp_pred, gp_sigma, _ = evaluate_gpr(gpr_pres, X_test_gpr_s, yp_test,
                                     scaler_yp_gpr, "GPR Pressure (Pa)")

# ========= 9. 模型性能汇总表 =========
print("\n" + "="*55)
print("===== 模型性能汇总 =====")
print("="*55)

summary_data = []
for label, y_true, y_pred in [
    ("RF-Velocity",    yv_true, rv_pred),
    ("RF-Temperature", yt_true, rt_pred),
    ("RF-Pressure",    yp_true, rp_pred),
    ("LGB-Velocity",   yv_true, lv_pred),
    ("LGB-Temperature",yt_true, lt_pred),
    ("LGB-Pressure",   yp_true, lp_pred),
    ("GPR-Velocity",   np.array(yv_test), gv_pred),
    ("GPR-Temperature",np.array(yt_test), gt_pred),
    ("GPR-Pressure",   np.array(yp_test), gp_pred),
]:
    rmse  = np.sqrt(mean_squared_error(y_true, y_pred))
    r2    = r2_score(y_true, y_pred)
    nrmse = rmse / (y_true.max() - y_true.min())
    summary_data.append({"Model": label, "R²": f"{r2:.4f}",
                          "RMSE": f"{rmse:.4e}", "NRMSE": f"{nrmse:.4%}"})

summary_df = pd.DataFrame(summary_data)
print(summary_df.to_string(index=False))
summary_df.to_csv(f"{DATA_DIR}\\model_comparison.csv", index=False)
print(f"\n汇总表已保存: model_comparison.csv")

# ========= 10. 特征重要性（RF） =========
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle("RF Feature Importance", fontsize=13)

for ax, model, title in [
    (axes[0], rf_vel,  "Velocity Model"),
    (axes[1], rf_temp, "Temperature Model"),
    (axes[2], rf_pres, "Pressure Model"),
]:
    imp     = model.feature_importances_
    indices = np.argsort(imp)[::-1]
    ax.bar(range(len(imp)), imp[indices], color="steelblue")
    ax.set_xticks(range(len(imp)))
    ax.set_xticklabels(np.array(FEATURES)[indices], rotation=30)
    ax.set_ylabel("Importance")
    ax.set_title(title)

plt.tight_layout()
plt.savefig(f"{DATA_DIR}\\feature_importance.png", dpi=150)
plt.show()

# ========= 11. 三模型 × 三物理量 预测vs真实 =========
fig, axes = plt.subplots(3, 3, figsize=(15, 13))
fig.suptitle("Predicted vs True  —  RF / LightGBM / Kriging", fontsize=14)

plot_cfg = [
    (rv_pred, lv_pred, gv_pred, yv_true, np.array(yv_test),
     "Velocity (m/s)",  "darkorange"),
    (rt_pred, lt_pred, gt_pred, yt_true, np.array(yt_test),
     "Temperature (K)", "steelblue"),
    (rp_pred, lp_pred, gp_pred, yp_true, np.array(yp_test),
     "Pressure (Pa)",   "seagreen"),
]
row_labels = ["RF", "LightGBM", "Kriging"]

for col, (rp, lp, gp, y_tr, y_gpr, label, color) in enumerate(plot_cfg):
    for row, (y_pred, y_true) in enumerate([
        (rp, y_tr), (lp, y_tr), (gp, y_gpr)
    ]):
        ax = axes[row, col]
        ax.scatter(y_true, y_pred, s=1, alpha=0.2, color=color)
        lo = min(y_true.min(), y_pred.min())
        hi = max(y_true.max(), y_pred.max())
        ax.plot([lo, hi], [lo, hi], "r--", linewidth=1)
        ax.set_xlabel(f"True {label}")
        ax.set_ylabel(f"Pred {label}")
        ax.set_title(f"{row_labels[row]} — {label}")

plt.tight_layout()
plt.savefig(f"{DATA_DIR}\\pred_vs_true.png", dpi=150)
plt.show()

# ========= 12. GPR不确定度分布图 =========
# Kriging独有：可视化预测不确定度，树模型无此能力
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle("Kriging Prediction Uncertainty (σ)", fontsize=13)

for ax, sigma, label, color in [
    (axes[0], gv_sigma, "Velocity (m/s)",  "darkorange"),
    (axes[1], gt_sigma, "Temperature (K)", "steelblue"),
    (axes[2], gp_sigma, "Pressure (Pa)",   "seagreen"),
]:
    ax.hist(sigma, bins=50, color=color, alpha=0.7, edgecolor="white")
    ax.set_xlabel(f"σ [{label}]")
    ax.set_ylabel("Count")
    ax.set_title(f"Uncertainty — {label}")
    ax.axvline(sigma.mean(), color="red", linestyle="--",
               label=f"mean={sigma.mean():.3e}")
    ax.legend()

plt.tight_layout()
plt.savefig(f"{DATA_DIR}\\gpr_uncertainty.png", dpi=150)
plt.show()

# ========= 13. 单工况流场对比（CFD vs RF vs GPR） =========
V_PLOT = 0.05
T_PLOT = 313.15
subset = data[
    (data["v_inlet"] == V_PLOT) & (data["T_inlet"] == T_PLOT)
].copy()

if len(subset) == 0:
    print(f"⚠ 找不到 v={V_PLOT}, T={T_PLOT} 的工况，跳过流场对比图")
else:
    X_sub       = subset[FEATURES]
    X_sub_s     = scaler_X_gpr.transform(X_sub)

    pred_vel_rf  = rf_vel.predict(X_sub)
    pred_temp_rf = rf_temp.predict(X_sub)

    pred_vel_gpr_s, sigma_vel = gpr_vel.predict(X_sub_s, return_std=True)
    pred_vel_gpr = scaler_yv_gpr.inverse_transform(
        pred_vel_gpr_s.reshape(-1, 1)).ravel()

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(
        f"Flow Field: CFD vs RF vs Kriging  |  v={V_PLOT} m/s, T={T_PLOT} K",
        fontsize=13)

    pairs = [
        (axes[0,0], subset["vel_mag"],  "coolwarm", "CFD Velocity (m/s)"),
        (axes[0,1], pred_vel_rf,        "coolwarm", "RF Predicted Velocity (m/s)"),
        (axes[0,2], pred_vel_gpr,       "coolwarm", "Kriging Predicted Velocity (m/s)"),
        (axes[1,0], subset["temperature"], "jet",   "CFD Temperature (K)"),
        (axes[1,1], pred_temp_rf,          "jet",   "RF Predicted Temperature (K)"),
        (axes[1,2], sigma_vel * scaler_yv_gpr.scale_[0],
                                           "Reds",  "Kriging Uncertainty σ (m/s)"),
    ]
    for ax, values, cmap, title in pairs:
        sc = ax.scatter(subset["x"], subset["y"],
                        c=values, cmap=cmap, s=1)
        plt.colorbar(sc, ax=ax)
        ax.set_title(title)
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")

    plt.tight_layout()
    plt.savefig(f"{DATA_DIR}\\field_comparison.png", dpi=150)
    plt.show()

# ========= 14. 保存模型 =========
joblib.dump(rf_vel,        f"{DATA_DIR}\\rf_vel.pkl")
joblib.dump(rf_temp,       f"{DATA_DIR}\\rf_temp.pkl")
joblib.dump(rf_pres,       f"{DATA_DIR}\\rf_pres.pkl")
joblib.dump(lgb_vel,       f"{DATA_DIR}\\lgb_vel.pkl")
joblib.dump(lgb_temp,      f"{DATA_DIR}\\lgb_temp.pkl")
joblib.dump(lgb_pres,      f"{DATA_DIR}\\lgb_pres.pkl")
joblib.dump(gpr_vel,       f"{DATA_DIR}\\gpr_vel.pkl")
joblib.dump(gpr_temp,      f"{DATA_DIR}\\gpr_temp.pkl")
joblib.dump(gpr_pres,      f"{DATA_DIR}\\gpr_pres.pkl")
joblib.dump(scaler_X_gpr,  f"{DATA_DIR}\\scaler_X_gpr.pkl")
joblib.dump(scaler_yv_gpr, f"{DATA_DIR}\\scaler_yv_gpr.pkl")
joblib.dump(scaler_yt_gpr, f"{DATA_DIR}\\scaler_yt_gpr.pkl")
joblib.dump(scaler_yp_gpr, f"{DATA_DIR}\\scaler_yp_gpr.pkl")
print("\n所有模型已保存至 DATA_DIR")
