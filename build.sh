#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="$SCRIPT_DIR/build"

echo "=========================================="
echo "  MVO Portfolio C++ 构建脚本"
echo "=========================================="

# 创建构建目录
mkdir -p "$BUILD_DIR"

# 检查依赖
EIGEN_DIR="$SCRIPT_DIR/thirdparty/eigen/eigen-3.4.0"
if [ -d "$EIGEN_DIR" ]; then
    echo "  [OK] Eigen3: $EIGEN_DIR"
else
    echo "  [ERROR] 未找到 Eigen3"
    exit 1
fi

NLOPT_LIB="$SCRIPT_DIR/thirdparty/nlopt/build/libnlopt.so"
if [ -f "$NLOPT_LIB" ]; then
    echo "  [OK] NLopt: $NLOPT_LIB"
else
    echo "  [ERROR] 未找到 NLopt"
    echo "  请先编译 thirdparty/nlopt"
    exit 1
fi

# CMake 配置
echo ""
echo "配置 CMake..."
cd "$BUILD_DIR"

cmake .. -DCMAKE_BUILD_TYPE=Release 2>&1

# 编译
echo ""
echo "编译..."
make -j$(nproc) 2>&1

echo ""
echo "=========================================="
echo "  构建完成!"
echo "=========================================="
echo ""
echo "运行验证程序:"
echo "  cd $BUILD_DIR"
echo "  ./algo_cpp_verify ../../data/data.csv"
echo ""
echo "运行主程序:"
echo "  ./algo_cpp ../../data/data.csv --strategy maxdiv --cov ew"
