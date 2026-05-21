## Model Performance Comparison

### Initial Results


[RF Velocity (m/s)]      Time: 0.1494s | RMSE: 7.7820e-04 | NRMSE: 0.3293% | R²: 0.999602
[RF Temperature (K)]    Time: 0.2422s | RMSE: 1.6963e-01 | NRMSE: 0.2262% | R²: 0.999943
[RF Pressure (Pa)]      Time: 0.2105s | RMSE: 4.4561e-01 | NRMSE: 0.0164% | R²: 0.999999

[LGB Velocity (m/s)]    Time: 0.1823s | RMSE: 4.2870e-04 | NRMSE: 0.1814% | R²: 0.999879
[LGB Temperature (K)]   Time: 0.1742s | RMSE: 1.8931e-01 | NRMSE: 0.2524% | R²: 0.999929
[LGB Pressure (Pa)]     Time: 0.1676s | RMSE: 1.0642e+00 | NRMSE: 0.0392% | R²: 0.999996

[MLP Velocity (m/s)]    Time: 0.6985s | RMSE: 3.7276e-02 | NRMSE: 410.1848% | R²: -616.454398
[MLP Temperature (K)]   Time: 0.2418s | RMSE: 7.8918e+03 | NRMSE: 467.0946% | R²: -241.440387
[MLP Pressure (Pa)]     Time: 0.2202s | RMSE: 2.9623e+05 | NRMSE: 21.7856% | R²: -0.344093


### Problem Analysis

The issue was caused by the absence of standardization on the test target data (`y_test`) during MLP evaluation. Specifically, the following transformation step was missing:

```python
yv_test_s = scaler_yv.transform(yv_test.values.reshape(-1,1)).ravel()
```

As a result, the MLP predictions and ground-truth values were evaluated under inconsistent scaling conditions, leading to abnormally large RMSE/NRMSE values and negative R² scores.

### Updated Results After Standardization

```text
[MLP Velocity (m/s)]    Time: 0.7754s | RMSE: 6.9867e-04 | NRMSE: 0.2957% | R²: 0.999679
[MLP Temperature (K)]   Time: 0.2748s | RMSE: 2.0182e-01 | NRMSE: 0.2691% | R²: 0.999920
[MLP Pressure (Pa)]     Time: 0.2289s | RMSE: 6.1423e+00 | NRMSE: 0.2264% | R²: 0.999855
```

After applying consistent target standardization, the MLP model achieved prediction accuracy comparable to RF and LGB models, with all R² values approaching 1.0 and NRMSE reduced to below 0.3%.

  优化部分：
35组CFD工况数据，采用5-fold CV训练RF代理模型：

vmax 预测 R² = 0.9993 ± 0.0007
tmax 预测 R² = 0.9843
NSGA-II 多目标优化耗时 16.50 s。

集成 inverse design 模块，支持基于目标温度约束的参数反求。

<img width="675" height="187.5" alt="pareto_front" src="https://github.com/user-attachments/assets/243216bb-42a3-4a2f-9990-e708bee3b473" />

<img width="525" height="187.5" alt="response_surface" src="https://github.com/user-attachments/assets/ba6b7d58-07e6-4f0a-a569-f2a8872d0f9f" />
