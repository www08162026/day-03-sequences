# D3 家庭用电预测：持续性基线与随机森林（student-work 副本）

本仓库是 Day 3 学生工作副本：按时间顺序比较"下一小时=上一小时"持续性基线与固定随机森林，预测下一小时平均功率，报告 MAE 与最大误差。

## 数据契约

- 数据所有者/发布者：UCI Machine Learning Repository（Kaggle 镜像：uciml）
- 标题：Individual household electric power consumption
- 原始 URL：https://www.kaggle.com/datasets/uciml/electric-power-consumption-data-set
- 许可：UCI 数据集许可，仅限课程用途
- 预期文件：`data/raw/household_power_consumption.txt`（2,075,259 条分钟记录、9 列）
- 使用边界：一户历史数据不能代表其他家庭，不用于电网控制或安全告警

## 环境与安装

```powershell
python --version
python -m pip install -r requirements.txt
```

## 运行路线（按顺序）

```powershell
# 1. 数据检查（必须先通过，失败就停止）
python analyze.py --check-data
# 预期：REAL DATA CHECK PASSED
# rows: 2075259
# columns: 9

# 2. 准备小时级课堂窗口（真实分钟数据的前 150,000 行）
python analyze.py --prepare --source-rows 150000
# 预期：REAL DATA PREPARATION PASSED；hourly_rows: 2501

# 3. 测试
python -m unittest discover -s tests -v
# 预期：Ran 2 tests ... OK

# 4. 主程序（保存 metrics.json、largest_errors.csv、forecast.png）
python analyze.py
```

## 结果文件

- `metrics.json`：时间范围、基线与候选的 MAE/RMSE、高需求时段（90% 分位）警告计数
- `largest_errors.csv`：绝对误差最大的 12 个小时
- `forecast.png`：留出区间首周的实际值 vs 基线与候选曲线

## 基线

持续性基线：下一小时预测 = 上一小时实测（`test["lag_1"]`）。不学习任何参数。

## 候选

`build_candidate()`：`RandomForestRegressor(n_estimators=100, random_state=2026)`。特征只用过去：`lag_1`、`lag_2`、`lag_3`、`lag_24`（小时功率）与 `hour_of_day`。严格按时间顺序划分（前 80% 训练、后 20% 测试）。

## 限制

- 训练/测试严格按时间先后，绝不随机打乱；
- 单户数据不推广到其他家庭，不用于电网控制；
- 报告和 PPT 中的数字都可以由上述命令重新产生。
