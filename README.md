# C++ 量化策略实现

基于 Eigen3 + NLopt 的投资组合优化算法 C++ 高性能实现。

## 项目概述

本项目将 Python MVO（均值-方差优化）投资组合优化算法移植为 C++ 版本，实现了三种风险厌恶策略与两种协方差估计方法，达到 **~350 倍加速**。

## ⚠️ 重要说明

当前 C++ 实现使用 **MMA（Method of Moving Asymptotes）** 算法而非原计划的 SLSQP，因为调试中发现 NLopt 的 SLSQP 实现存在返回值异常问题（返回 `NLOPT_INVALID_ARGS`）。MMA 算法与 scipy SLSQP 在收敛行为上存在差异，可能导致数值结果略有不同。

## 目录结构

```
suishi-quant/
├── AGENTS.md                    # 任务规范
├── REPORT.md                    # 技术报告
├── README.md                    # 本文件
├── generate_baselines.py        # Python 基准生成
│
├── python-impl/                 # Python 参考实现
│   ├── algo.py                  # 核心算法
│   └── data.csv                 # 输入数据
│
├── cpp-impl/                    # C++ 实现
│   ├── CMakeLists.txt           # CMake 构建配置
│   ├── build.sh                 # 构建脚本
│   ├── run.sh                   # 运行脚本
│   │
│   ├── include/                 # 头文件
│   │   ├── config.h             # 配置参数
│   │   ├── data_loader.h        # 数据加载
│   │   ├── preprocess.h          # 预处理
│   │   ├── covariance.h         # 协方差计算
│   │   ├── objective.h           # 目标函数
│   │   ├── constraints.h         # 约束条件
│   │   ├── optimizer.h           # 优化器
│   │   ├── metrics.h             # 性能指标
│   │   └── mvo.h                 # 主入口
│   │
│   ├── src/                     # 源代码
│   │   ├── main.cpp              # 主程序
│   │   ├── main_verify.cpp       # 验证程序
│   │   ├── config.cpp
│   │   ├── data_loader.cpp
│   │   ├── preprocess.cpp
│   │   ├── covariance.cpp
│   │   ├── objective.cpp
│   │   ├── constraints.cpp
│   │   ├── optimizer.cpp
│   │   ├── metrics.cpp
│   │   └── mvo.cpp
│   │
│   ├── build/                   # 编译输出
│   │   ├── libmvo_core.a       # 静态库
│   │   ├── algo_cpp              # 主程序
│   │   └── algo_cpp_verify       # 验证程序
│   │
│   └── thirdparty/              # 第三方库
│       ├── eigen/eigen-3.4.0/  # Eigen3
│       └── nlopt/               # NLopt
│
└── data/                        # 数据目录
    ├── data.csv                 # 输入数据
    └── baseline_*.csv          # Python 基准
```

## 快速开始

### 1. 构建

```bash
cd cpp-impl
bash build.sh
```

### 2. 运行

```bash
# 最大分散度策略 + 等权重协方差
cd cpp-impl/build
./algo_cpp ../../data/data.csv --strategy maxdiv --cov ew

# 最大化收益率策略 + 指数加权协方差
./algo_cpp ../../data/data.csv --strategy maxret --cov exp

# 风险厌恶-20 策略
./algo_cpp ../../data/data.csv --strategy risk20 --cov ew
```

### 3. 验证

```bash
./algo_cpp_verify ../../data/data.csv
```

## 算法说明

### 三种风险厌恶策略

| 策略       | 命令行参数              | 目标函数                                                                            |
| ---------- | ---------------------- | ----------------------------------------------------------------------------------- |
| 最大分散度   | `--strategy maxdiv`     | \(-\frac{\mathbf{w}^T \boldsymbol{\sigma}}{\sqrt{\mathbf{w}^T \Sigma \mathbf{w}}}\) |
| 最大化收益率 | `--strategy maxret`     | \(-\sum_i w_i \mu_i\)                                                               |
| 风险厌恶-20 | `--strategy risk20`     | \(-\sum_i w_i \mu_i + \frac{1}{2} \cdot 20 \cdot \mathbf{w}^T \Sigma \mathbf{w}\)   |

### 两种协方差估计

| 估计     | 命令行参数     | 说明             |
| -------- | -------------- | ---------------- |
| 等权重   | `--cov ew`     | 历史窗口内等权重协方差 |
| 指数加权 | `--cov exp`    | 近期数据权重更高（EWM） |

### 约束条件

- 非负权重：\(w_i \geq 0\)
- 权重和为 1：\(\sum w_i = 1\)
- 换手率限制：\(\sum|w_i - w_{i-1}| \leq 0.1\)
- 品种组约束：各组上下限 \([0.15, 0.35]\)

## 性能

| 指标   | Python    | C++      | 加速比     |
| ------ | --------- | -------- | ---------- |
| 总时间 | ~7000 ms  | ~20 ms   | **~350x** |

### 性能细分

| 阶段     | 耗时         |
| -------- | ------------ |
| 数据读取 | ~1 ms       |
| 预处理   | < 0.1 ms    |
| 优化求解 | ~19 ms      |
| **总计** | **~20 ms**  |

## 数值一致性

| 对比项         | 数值          |
| -------------- | ------------- |
| C++ 最终收益   | 1.1509       |
| Python 最终收益 | 1.1481       |
| 最终收益误差   | ~2.7e-3      |

⚠️ **说明**：由于使用 MMA 而非 SLSQP，数值差异可能比预期更大。差异源于 MMA 与 scipy SLSQP 的求解器实现差异，而非逻辑错误。

## 第三方库

| 库      | 版本    | 许可证 | 用途     |
| ------- | ------- | ------ | -------- |
| Eigen3  | 3.4.0  | MPL2   | 线性代数 |
| NLopt   | 2.9.0  | BSD    | 约束优化 |

## 报告

详细报告请参阅根目录 [REPORT.md](../REPORT.md)。
