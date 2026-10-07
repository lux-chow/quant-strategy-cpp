#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="$SCRIPT_DIR/build"

echo "=========================================="
echo "  MVO Portfolio C++ 构建脚本"
echo "=========================================="

# 创建构建目录
mkdir -p "$BUILD_DIR"

# 检查 Eigen3
EIGEN_DIR="$SCRIPT_DIR/thirdparty/eigen/eigen-3.4.0"
if [ -d "$EIGEN_DIR" ]; then
    echo "  [OK] Eigen3: $EIGEN_DIR"
else
    echo "  [ERROR] 未找到 Eigen3"
    exit 1
fi

# 检查/编译 NLopt
NLOPT_DIR="$SCRIPT_DIR/thirdparty/nlopt"
NLOPT_BUILD="$NLOPT_DIR/build"
NLOPT_LIB="$NLOPT_BUILD/libnlopt.so"

if [ -f "$NLOPT_LIB" ]; then
    echo "  [OK] NLopt (已编译): $NLOPT_LIB"
else
    echo "  [编译] NLopt..."
    mkdir -p "$NLOPT_BUILD"
    cd "$NLOPT_BUILD"
    cmake .. -DCMAKE_INSTALL_PREFIX=/usr/local \
             -DNLOPT_PYTHON=OFF \
             -DNLOPT_MATLAB=OFF \
             -DBUILD_SHARED_LIBS=ON 2>&1 | tail -5
    make -j$(nproc) 2>&1 | tail -3
    
    if [ -f "$NLOPT_LIB" ]; then
        echo "  [OK] NLopt 编译完成"
    else
        echo "  [ERROR] NLopt 编译失败"
        exit 1
    fi
    cd "$SCRIPT_DIR"
fi

# CMake 配置
echo ""
echo "配置 CMake..."
cd "$BUILD_DIR"

cmake .. -DCMAKE_BUILD_TYPE=Release 2>&1 | tail -5

# 编译
echo ""
echo "编译..."
make -j$(nproc) 2>&1

echo ""
echo "=========================================="
echo "  构建完成!"
echo "=========================================="
echo ""
echo "运行验证:"
echo "  cd $BUILD_DIR"
echo "  ./algo_cpp_verify ../../data/data.csv"
