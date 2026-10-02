# C++ 量化策略实现

基于 Eigen3 + NLopt 的投资组合优化算法 C++ 实现。

## 目录结构

```
cpp-impl/
├── CMakeLists.txt            # CMake 构建配置
├── build.sh                  # 构建脚本
├── run.sh                    # 运行脚本
├── README.md
├── REPORT.md                # 详细报告
│
├── include/                  # 头文件（模块化接口）
│   ├── mvo.h               # 库入口
│   ├── config.h             # 配置参数
│   ├── data_loader.h       # 数据加载
│   ├── preprocess.h         # 预处理（3σ 极值处理、rolling window）
│   ├── covariance.h         # 协方差计算
│   ├── objective.h          # 目标函数
│   ├── constraints.h        # 约束条件
│   ├── optimizer.h          # 优化器封装
│   └── metrics.h           # 性能指标
│
├── src/                     # 源代码
│   ├── main.cpp            # 主程序（输出技术指标）
│   ├── main_verify.cpp     # 验证程序（输出详细权重）
│   ├── config.cpp
│   ├── data_loader.cpp
│   ├── preprocess.cpp
│   ├── covariance.cpp
│   ├── objective.cpp
│   ├── constraints.cpp
│   ├── optimizer.cpp
│   └── metrics.cpp
│
├── tests/                   # 单元测试
│   ├── tests.h
│   └── tests_main.cpp
│
├── build/                   # 编译输出目录
│   ├── algo_cpp             # 主程序
│   ├── algo_cpp_tests      # 测试程序
│   ├── algo_cpp_verify     # 验证程序
│   └── libmvo_core.a        # 静态库
│
├── lib/local/               # 编译后的本地库
│   ├── include/             # 头文件 (Eigen, nlopt)
│   └── lib/                 # 共享库 (libnlopt.so)
│
└── thirdparty/              # 第三方库源码
    ├── eigen/               # Eigen3 线性代数库
    ├── nlopt/               # NLopt 非线性优化库
    └── cmake/               # CMake 构建工具
```

根目录文件：

```
suishi-quant/
├── AGENTS.md               # 任务说明
├── generate_baselines.py   # Python 基准生成
└── data/                    # 数据文件
    ├── data.csv
    ├── baseline_w_*.csv     # Python 基准权重
    ├── baseline_r_*.csv     # Python 基准收益
    ├── w_*.csv               # C++ 权重
    └── r_*.csv               # C++ 加权收益
    # 命名规则：
    #   prefix: baseline_w | baseline_r | w | r
    #   strategy: maxdiv | maxret | risk20
    #   cov:      ew (等权重) | exp (指数加权)
```

## 快速开始

### 1. 构建

```bash
cd cpp-impl
bash build.sh
```

### 2. 运行

```bash
cd cpp-impl/build
./algo_cpp_verify ../../data/data.csv
```

### 3. 运行测试

```bash
cd cpp-impl/build
./algo_cpp_tests
```

### 4. Python 基准生成

```bash
cd ..
./venv/bin/python generate_baselines.py
```

## 模块说明

| 模块 | 职责 |
|------|------|
| `config` | 配置参数集中管理 |
| `data_loader` | CSV 数据读取 |
| `preprocess` | 3σ 极值处理、rolling window 构建 |
| `covariance` | 等权重/指数加权协方差计算 |
| `objective` | 三种目标函数（最大分散度、均值-方差） |
| `constraints` | 十一约束条件（SLSQP 格式） |
| `optimizer` | NLopt SLSQP 封装 |
| `metrics` | 夏普比、最大回撤等性能指标 |

## 算法说明

### 三种风险厌恶策略

| 策略 | 代码 | 说明 |
|------|------|------|
| 最大分散 | `maxdiv` | 最大化分散度比 |
| 最大目标收益率 | `maxret` | 最大化期望收益 |
| 风险厌恶-20 | `risk20` | 均值-方差优化 |

### 两种协方差估计

| 估计 | 代码 | 说明 |
|------|------|------|
| 等权重 | `ew` | 历史窗口内等权重 |
| 指数加权 | `exp` | 近期数据权重更高（递归 EMA） |

## 性能

| 指标 | Python | C++ | 加速比 |
|------|--------|------|--------|
| 总时间 | ~7000ms | ~32ms | **~219x** |

### 性能细分

| 阶段 | 耗时 |
|------|------|
| 数据读取 | 0.4 ms |
| 预处理 | 0.08 ms |
| 优化求解 | 31 ms |
| 输出 | 0.03 ms |
| **总计** | **~32 ms** |

## 数值一致性说明

⚠️ **注意**：由于 NLopt LD_SLSQP 与 scipy.optimize SLSQP 的内部实现差异，C++ 数值结果与 Python 存在误差。

**已修复的问题**：
- ✅ EWM maxdiv 使用标准差而非方差
- ✅ EWM 均值使用递归 EMA

**差异来源**：
- 求解器内部算法实现不同
- 收敛准则和搜索策略差异

详见 [REPORT.md](REPORT.md)。

## 第三方库

| 库 | 版本 | 许可证 |
|----|------|--------|
| Eigen3 | 3.4.x | MPL2 |
| NLopt | 2.7.x | BSD |

## 报告

详细报告请参阅 [REPORT.md](REPORT.md)。
# quant-strategy-cpp
