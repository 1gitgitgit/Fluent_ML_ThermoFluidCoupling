import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import joblib
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import lightgbm as lgb

# ========= 1. 配置 =========
DATA_DIR     = r"D:\VScode\project\2026\demo1\2026-4-24\py\data"
VELOCITIES   = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1]
TEMPERATURES = [293.15, 303.15, 313.15, 323.15, 353.15]

# ========= 2. 水的运动粘度 ν(T) 多项式拟合 =========
_T_known  = np.array([293.15, 303.15, 313.15, 323.15, 353.15])
_nu_known = np.array([1.004e-6, 0.801e-6, 0.658e-6, 0.553e-6, 0.365e-6])
_nu_poly  = np.polyfit(_T_known, _nu_known, 3)

def nu_water(T):
    return np.polyval(_nu_poly, T)

# ========= 3. 读取数据 =========
dfs = []
for v in VELOCITIES:
    for T in TEMPERATURES:
        fname = f"v{v:.4f}_T{T:.1f}.csv"
        fpath = f"{DATA_DIR}\\{fname}"
        try:
            df = pd.read_csv(fpath, sep=",", skiprows=1, header=None)
            df.columns = ["cellnumber", "x", "y", "x2", "y2",
                          "vel_mag", "temperature", "pressure"]
            original_count = len(df)
            df = df.replace([np.inf, -np.inf], np.nan).dropna()
            dropped   = original_count - len(df)
            drop_rate = dropped / original_count
            print(f"工况 v={v}, T={T}: {len(df)} 个点, 清洗 {dropped} 个 ({drop_rate:.2%})")
            if drop_rate > 0.05:
                print(f"  ❗ 警告: 剔除比例过高 ({drop_rate:.2%})")
            df["v_inlet"] = v
            df["T_inlet"] = T
            df = df.drop(columns=["cellnumber", "x2", "y2"])
            dfs.append(df)
        except FileNotFoundError:
            print(f"  ⚠ 文件不存在，跳过: {fname}")

data = pd.concat(dfs, ignore_index=True)

# ========= 4. 特征工程：加入 Re =========
L = data["x"].max() - data["x"].min()
data["Re"] = data["v_inlet"] * L / nu_water(data["T_inlet"])
print(f"\n特征长度 L = {L:.4f} m")
print(f"Re 范围: {data['Re'].min():.1f} ~ {data['Re'].max():.1f}")
print(f"总样本数: {len(data)}")

# ========= 5. 特征 & 标签 =========
FEATURES = ["x", "y", "v_inlet", "T_inlet", "Re"]
X      = data[FEATURES]
y_vel  = data["vel_mag"]
y_temp = data["temperature"]
y_pres = data["pressure"]

# ========= 6. 按边界工况划分（留出 v=0.1 OR T=353.15 共11个工况） =========
is_boundary = (data["v_inlet"] == 0.1) | (data["T_inlet"] == 353.15)
train_mask  = ~is_boundary
test_mask   = is_boundary

X_train  = X[train_mask];  X_test  = X[test_mask]
yv_train = y_vel[train_mask];  yv_test = y_vel[test_mask]
yt_train = y_temp[train_mask]; yt_test = y_temp[test_mask]
yp_train = y_pres[train_mask]; yp_test = y_pres[test_mask]

n_test_cases = data[test_mask][["v_inlet","T_inlet"]].drop_duplicates().shape[0]
print(f"\n训练集(内部工况): {len(X_train)}  测试集(边界工况): {len(X_test)}")
print(f"测试工况数: {n_test_cases}/35")

# ========= 7. 评估函数 =========
def evaluate(model, X_test, y_test, name, scaler_y=None):
    t0     = time.time()
    y_pred = model.predict(X_test)
    t1     = time.time()
    if scaler_y is not None:
        y_pred = scaler_y.inverse_transform(y_pred.reshape(-1,1)).ravel()
        y_test = scaler_y.inverse_transform(np.array(y_test).reshape(-1,1)).ravel()
    rmse  = np.sqrt(mean_squared_error(y_test, y_pred))
    r2    = r2_score(y_test, y_pred)
    nrmse = rmse / (y_test.max() - y_test.min())
    print(f"  [{name}]  耗时: {t1-t0:.4f}s | RMSE: {rmse:.4e} | NRMSE: {nrmse:.4%} | R²: {r2:.6f}")
    return y_pred, np.array(y_test)

# ========= 8. Grid Search（仅对速度目标，最优参数复用至温度/压力） =========
print("\n" + "="*50)
print("===== Grid Search 调参（基于 Velocity 目标）=====")
print("="*50)

# --- RF ---
rf_gs = GridSearchCV(
    RandomForestRegressor(n_jobs=-1, random_state=42),
    {"n_estimators": [100, 200], "max_depth": [10, 15, 20], "min_samples_leaf": [1, 3]},
    cv=3, scoring="neg_root_mean_squared_error", n_jobs=-1, verbose=1
)
print("\nRF GridSearch...")
t0 = time.time()
rf_gs.fit(X_train, yv_train)
best_rf_params = rf_gs.best_params_
print(f"  最优参数: {best_rf_params}  耗时: {time.time()-t0:.1f}s")

# --- LGB ---
lgb_gs = GridSearchCV(
    lgb.LGBMRegressor(n_jobs=-1, random_state=42, verbose=-1),
    {"num_leaves": [31, 63], "learning_rate": [0.05, 0.1], "n_estimators": [300, 500]},
    cv=3, scoring="neg_root_mean_squared_error", n_jobs=-1, verbose=1
)
print("\nLGB GridSearch...")
t0 = time.time()
lgb_gs.fit(X_train, yv_train)
best_lgb_params = lgb_gs.best_params_
print(f"  最优参数: {best_lgb_params}  耗时: {time.time()-t0:.1f}s")

# --- MLP（Pipeline内做标准化，保证CV不泄漏） ---
mlp_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("mlp", MLPRegressor(activation="relu", solver="adam", max_iter=1000,
                          early_stopping=True, validation_fraction=0.1,
                          n_iter_no_change=30, random_state=42, verbose=False))
])
mlp_gs = GridSearchCV(
    mlp_pipe,
    {"mlp__hidden_layer_sizes": [(128, 128, 64), (256, 256, 128, 64)],
     "mlp__learning_rate_init": [1e-3, 5e-4]},
    cv=3, scoring="neg_root_mean_squared_error", n_jobs=1, verbose=1
)
print("\nMLP GridSearch...")
t0 = time.time()
mlp_gs.fit(X_train, yv_train)
best_mlp_params = {k.replace("mlp__", ""): v for k, v in mlp_gs.best_params_.items()}
print(f"  最优参数: {best_mlp_params}  耗时: {time.time()-t0:.1f}s")

# ========= 9. Random Forest =========
print("\n" + "="*50)
print("===== Random Forest（最优参数）=====")
print("="*50)

rf_vel  = RandomForestRegressor(**best_rf_params, n_jobs=-1, random_state=42)
rf_temp = RandomForestRegressor(**best_rf_params, n_jobs=-1, random_state=42)
rf_pres = RandomForestRegressor(**best_rf_params, n_jobs=-1, random_state=42)

total_t = time.time()
for model, ytr, name in [
    (rf_vel,  yv_train, "RF-Velocity"),
    (rf_temp, yt_train, "RF-Temperature"),
    (rf_pres, yp_train, "RF-Pressure"),
]:
    t0 = time.time()
    print(f"  训练 {name}...")
    model.fit(X_train, ytr)
    print(f"  耗时: {time.time()-t0:.2f}s")
print(f"RF 总训练耗时: {time.time()-total_t:.2f}s\n")

rv_pred, yv_true = evaluate(rf_vel,  X_test, yv_test, "RF Velocity (m/s)")
rt_pred, yt_true = evaluate(rf_temp, X_test, yt_test, "RF Temperature (K)")
rp_pred, yp_true = evaluate(rf_pres, X_test, yp_test, "RF Pressure (Pa)")

# ========= 10. LightGBM =========
print("\n" + "="*50)
print("===== LightGBM（最优参数）=====")
print("="*50)

lgb_vel  = lgb.LGBMRegressor(**best_lgb_params, n_jobs=-1, random_state=42, verbose=-1)
lgb_temp = lgb.LGBMRegressor(**best_lgb_params, n_jobs=-1, random_state=42, verbose=-1)
lgb_pres = lgb.LGBMRegressor(**best_lgb_params, n_jobs=-1, random_state=42, verbose=-1)

total_t = time.time()
for model, ytr, name in [
    (lgb_vel,  yv_train, "LGB-Velocity"),
    (lgb_temp, yt_train, "LGB-Temperature"),
    (lgb_pres, yp_train, "LGB-Pressure"),
]:
    t0 = time.time()
    print(f"  训练 {name}...")
    model.fit(X_train, ytr)
    print(f"  耗时: {time.time()-t0:.2f}s")
print(f"LightGBM 总训练耗时: {time.time()-total_t:.2f}s\n")

lv_pred, _ = evaluate(lgb_vel,  X_test, yv_test, "LGB Velocity (m/s)")
lt_pred, _ = evaluate(lgb_temp, X_test, yt_test, "LGB Temperature (K)")
lp_pred, _ = evaluate(lgb_pres, X_test, yp_test, "LGB Pressure (Pa)")

# ========= 11. MLP =========
print("\n" + "="*50)
print("===== MLP（最优参数）=====")
print("="*50)

scaler_X  = StandardScaler()
scaler_yv = StandardScaler()
scaler_yt = StandardScaler()
scaler_yp = StandardScaler()

X_train_s  = scaler_X.fit_transform(X_train)
X_test_s   = scaler_X.transform(X_test)
yv_train_s = scaler_yv.fit_transform(yv_train.values.reshape(-1,1)).ravel()
yt_train_s = scaler_yt.fit_transform(yt_train.values.reshape(-1,1)).ravel()
yp_train_s = scaler_yp.fit_transform(yp_train.values.reshape(-1,1)).ravel()
yv_test_s  = scaler_yv.transform(yv_test.values.reshape(-1,1)).ravel()
yt_test_s  = scaler_yt.transform(yt_test.values.reshape(-1,1)).ravel()
yp_test_s  = scaler_yp.transform(yp_test.values.reshape(-1,1)).ravel()

mlp_vel  = MLPRegressor(**best_mlp_params, activation="relu", solver="adam",
                         max_iter=1000, early_stopping=True, validation_fraction=0.1,
                         n_iter_no_change=30, random_state=42, verbose=False)
mlp_temp = MLPRegressor(**best_mlp_params, activation="relu", solver="adam",
                         max_iter=1000, early_stopping=True, validation_fraction=0.1,
                         n_iter_no_change=30, random_state=42, verbose=False)
mlp_pres = MLPRegressor(**best_mlp_params, activation="relu", solver="adam",
                         max_iter=1000, early_stopping=True, validation_fraction=0.1,
                         n_iter_no_change=30, random_state=42, verbose=False)

total_t = time.time()
for model, ytr_s, name in [
    (mlp_vel,  yv_train_s, "MLP-Velocity"),
    (mlp_temp, yt_train_s, "MLP-Temperature"),
    (mlp_pres, yp_train_s, "MLP-Pressure"),
]:
    t0 = time.time()
    print(f"  训练 {name}...")
    model.fit(X_train_s, ytr_s)
    print(f"  耗时: {time.time()-t0:.2f}s  实际迭代: {model.n_iter_} 轮")
print(f"MLP 总训练耗时: {time.time()-total_t:.2f}s\n")

mv_pred, yv_mlp = evaluate(mlp_vel,  X_test_s, yv_test_s, "MLP Velocity (m/s)",  scaler_yv)
mt_pred, yt_mlp = evaluate(mlp_temp, X_test_s, yt_test_s, "MLP Temperature (K)", scaler_yt)
mp_pred, yp_mlp = evaluate(mlp_pres, X_test_s, yp_test_s, "MLP Pressure (Pa)",   scaler_yp)

# ========= 12. 特征重要性（RF） =========
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

# ========= 13. 预测 vs 真实 =========
fig, axes = plt.subplots(3, 3, figsize=(15, 13))
fig.suptitle("Predicted vs True  —  RF / LightGBM / MLP\n(边界工况外推测试)", fontsize=14)
plot_cfg = [
    (rv_pred, lv_pred, mv_pred, yv_true, yv_mlp, "Velocity (m/s)",  "darkorange"),
    (rt_pred, lt_pred, mt_pred, yt_true, yt_mlp, "Temperature (K)", "steelblue"),
    (rp_pred, lp_pred, mp_pred, yp_true, yp_mlp, "Pressure (Pa)",   "seagreen"),
]
row_labels = ["RF", "LightGBM", "MLP"]
for col, (rp, lp, mp, y_tr, y_mlp, label, color) in enumerate(plot_cfg):
    for row, (y_pred, y_true) in enumerate([(rp, y_tr), (lp, y_tr), (mp, y_mlp)]):
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

# ========= 14. 单工况流场对比（RF） =========
V_PLOT, T_PLOT = 0.05, 313.15
subset = data[(data["v_inlet"] == V_PLOT) & (data["T_inlet"] == T_PLOT)].copy()
if len(subset) == 0:
    print(f"⚠ 找不到 v={V_PLOT}, T={T_PLOT} 的工况，跳过")
else:
    X_sub     = subset[FEATURES]
    pred_vel  = rf_vel.predict(X_sub)
    pred_temp = rf_temp.predict(X_sub)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Flow Field Comparison  v={V_PLOT} m/s, T={T_PLOT} K", fontsize=13)
    for ax, values, cmap, title in [
        (axes[0,0], subset["vel_mag"],     "coolwarm", "CFD Velocity (m/s)"),
        (axes[0,1], pred_vel,              "coolwarm", "RF Predicted Velocity (m/s)"),
        (axes[1,0], subset["temperature"], "jet",      "CFD Temperature (K)"),
        (axes[1,1], pred_temp,             "jet",      "RF Predicted Temperature (K)"),
    ]:
        sc = ax.scatter(subset["x"], subset["y"], c=values, cmap=cmap, s=1)
        plt.colorbar(sc, ax=ax)
        ax.set_title(title); ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    plt.tight_layout()
    plt.savefig(f"{DATA_DIR}\\field_comparison.png", dpi=150)
    plt.show()

# ========= 15. 保存 =========
for obj, fname in [
    (rf_vel,   "rf_vel"),   (rf_temp,  "rf_temp"),  (rf_pres,  "rf_pres"),
    (lgb_vel,  "lgb_vel"),  (lgb_temp, "lgb_temp"), (lgb_pres, "lgb_pres"),
    (scaler_X, "scaler_X"), (scaler_yv,"scaler_yv"),(scaler_yt,"scaler_yt"),
    (scaler_yp,"scaler_yp"),
]:
    joblib.dump(obj, f"{DATA_DIR}\\{fname}.pkl")
print("\n所有模型已保存至 DATA_DIR")