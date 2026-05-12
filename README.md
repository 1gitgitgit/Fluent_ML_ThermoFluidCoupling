# Fluent_ML_ThermoFluidCoupling
Fluent_ML_Thermo-fluid coupling_Reverse optimization对一个简单的圆柱绕流进行流热耦合与ml和反向优化
## 模型性能对比表

| 模型 | 变量 | 耗时 (s) | RMSE | NRMSE (%) | R² |
|------|------|----------|------|-----------|-----|
| RF | Velocity (m/s) | 0.1494 | 7.7820e-04 | 0.3293 | 0.999602 |
| RF | Temperature (K) | 0.2422 | 1.6963e-01 | 0.2262 | 0.999943 |
| RF | Pressure (Pa) | 0.2105 | 4.4561e-01 | 0.0164 | 0.999999 |
| LGB | Velocity (m/s) | 0.1823 | 4.2870e-04 | 0.1814 | 0.999879 |
| LGB | Temperature (K) | 0.1742 | 1.8931e-01 | 0.2524 | 0.999929 |
| LGB | Pressure (Pa) | 0.1676 | 1.0642e+00 | 0.0392 | 0.999996 |
| MLP | Velocity (m/s) | 0.6985 | 3.7276e-02 | 410.1848 | -616.454398 |
| MLP | Temperature (K) | 0.2418 | 7.8918e+03 | 467.0946 | -241.440387 |
| MLP | Pressure (Pa) | 0.2202 | 2.9623e+05 | 21.7856 | -0.344093 |

<img width="563" height="150" alt="feature_importance" src="https://github.com/user-attachments/assets/a2e14f1d-23f9-4879-ad2d-5fb040ea6914" />
<img width="562.5" height="487.5" alt="pred_vs_true" src="https://github.com/user-attachments/assets/eccc8633-a9d2-4fab-89c4-ea9f496675da" />
<img width="563" height="375" alt="field_comparison" src="https://github.com/user-attachments/assets/e724c822-f6de-48f2-bb8d-d4c7abf34fb5" />



