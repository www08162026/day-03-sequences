# 每日作业报告

## 1. 本日问题

- 里程碑：day-03
- 学生或小组：LHL
- 使用者：希望提前看到用电变化、但不能依赖昂贵系统的家庭能源研究小组
- 真实输入：UCI 家庭用电数据前 150,000 条真实分钟记录（2006-12 至 2007-03），聚合为 2,501 个小时平均值
- 需要的输出：持续性基线与随机森林在同一后段测试区间上的 MAE、RMSE 与高需求时段警告对比
- 与使用者最相关的错误：高需求时段的漏报（真实高用电小时被预测成低用电）
- 本日产品边界：只预测下一小时平均功率，不推广到其他家庭，不用于电网控制

## 2. 真实数据或真实课程输入

- 所有者/发布者：UCI Machine Learning Repository（Kaggle 镜像发布者 uciml）
- 标题：Individual household electric power consumption
- 原始 URL：https://www.kaggle.com/datasets/uciml/electric-power-consumption-data-set
- 许可标签或使用许可：UCI 数据集许可，仅限本课程用途
- 下载/取得日期：2026-08-17
- 预期文件与结构：`data/raw/household_power_consumption.txt`，2,075,259 条分钟记录、9 列
- 检查命令：`python analyze.py --check-data`；`python analyze.py --prepare --source-rows 150000`
- 实际检查结果：`REAL DATA CHECK PASSED`；rows: 2075259；columns: 9。准备后 hourly_rows: 2501，时间范围 2006-12-16 17:00 至 2007-03-30 21:00
- 已知缺失、偏差或限制：原文件含 `?` 缺失值已删除；只使用单户数据；150,000 行窗口不代表全年分布

## 3. 可复现运行

```powershell
# 当前目录
ai-camp-2026-deploy\day-03-sequences

# 安装
python -m pip install -r requirements.txt

# 数据检查
python analyze.py --check-data

# 数据准备
python analyze.py --prepare --source-rows 150000

# 测试
python -m unittest discover -s tests -v

# 主程序
python analyze.py
```

关键预期输出：数据检查 `REAL DATA CHECK PASSED`（2,075,259 行）；测试 `Ran 2 tests ... OK`；主程序写 `metrics.json`、`largest_errors.csv` 和 `forecast.png`。

## 4. 基线与候选

### 简单基线

- 方法：持续性基线，下一小时预测值 = 上一小时实测值（`test["lag_1"]`）
- 为什么足够简单：不学习任何参数，只复制最近一小时
- 命令：`python analyze.py`
- 结果：MAE 0.794 kW、RMSE 1.112 kW；高需求警告（阈值 3.16 kW，训练集 90% 分位）召回 0.184，漏报 31 次

### 候选方法

- 学生完成的核心改动：`make_lagged`（滞后特征与下一小时目标）和 `build_candidate`（固定种子 RandomForestRegressor）；另按 starter README 文档补上 `--check-data` 完整数据检查
- 保持不变的条件：同一 2,501 小时窗口、同一 80/20 时间顺序划分、同一 LAGS=(1,2,3,24)、同一指标
- 命令：`python analyze.py`
- 结果：MAE 0.577 kW、RMSE 0.791 kW；高需求警告召回 0.132，漏报 33 次

| 项目 | 基线 | 候选 | 含义 |
| --- | ---: | ---: | --- |
| MAE (kW) | 0.794 | 0.577 | 平均绝对误差下降 27% |
| RMSE (kW) | 1.112 | 0.791 | 大误差同时被压缩 |
| 高需求漏报 | 31 | 33 | 平均误差下降但突升时段仍常漏报 |

## 5. 一个真实失败案例

- 样本位置/编号：`largest_errors.csv` 第 1 行，timestamp=2007-03-28 17:00
- 真实结果：0.386 kW（用电骤降的傍晚）
- 系统输出：预测 3.442 kW（绝对误差 3.057 kW）
- 可以观察到什么：模型按近期高功率和滞后特征推断傍晚仍高用电，但该日实际骤降
- 说明的限制：随机森林只能从过去小时模式外推，无法预见单日行为突变（如离家、断电）
- 不能证明什么：不能证明预测错误的原因是"该家庭离开家"，缺少该日任何外部事件记录
- 下一项最小检查：检查 2007-03-28 当天滞后 24 小时前后的功率模式，判断是否与每周作息冲突

## 6. 智能体与学生工作边界

- 智能体提出/生成/修改了什么：智能体实现了 `make_lagged`、`build_candidate` 两个 TODO，并补充了 README 文档化的 `--check-data` 数据检查命令
- 学生怎样核对文件、来源、输出、测试和 diff：运行 `--check-data` 和 `--prepare` 核对行数；运行测试确认 shift 语义（lag_24=0、target_next=25）；打开 `metrics.json` 对照数字；用 `git diff` 确认改动集中在三个函数
- 学生修改或拒绝了什么建议：拒绝用随机划分测试集（会破坏时间顺序）；拒绝加入未来信息特征（目标泄漏）；拒绝调参换取更高分数
- 每名成员能独立解释的代码或证据：`make_lagged` 的 shift 方向、`chronological_split` 的保序划分、`warning_counts` 的四个警告计数

## 7. 结论与限制

1. 随机森林候选把 MAE 从 0.794 降到 0.577 kW（-27%），RMSE 从 1.112 降到 0.791 kW。2. 两种模型在同一后段 496 小时测试区间、同一时间顺序划分上比较，条件一致。3. 高需求时段（≥3.16 kW）漏报仍达 33 次，候选召回率 0.132 甚至略低于基线 0.184，说明平均误差改善没有转化为突增提示的可靠性。4. 最大误差 3.06 kW 出现在用电骤降时刻，说明模型对行为突变无能为力。5. 数据限制：只使用一户 2006-2007 冬季窗口，不推广到其他家庭或季节。6. 方法限制：仅滞后特征无法建模外部事件，随机森林不能外推未见过的功率水平。7. 不能用于真实决策：不能用于电网控制、安全告警或与其他家庭对比。

## 8. 提交复核

- [x] README 从新环境可以开始运行
- [x] 数据检查、测试和主程序重新运行
- [x] 报告数字与保存输出一致
- [x] `presentation.pptx` 在 3 分钟内讲完
- [x] `submission.json` 路径正确
- [x] 无密钥、大数据、私人信息、虚拟环境或缓存
- [ ] GitHub 网页复查并邮件发送 URL（由学生本人完成）
