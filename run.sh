#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="$SCRIPT_DIR/build"

if [ ! -f "$BUILD_DIR/algo_cpp" ]; then
    echo "错误: 未找到 algo_cpp"
    echo "请先运行: bash build.sh"
    exit 1
fi

# 默认参数
DATA_PATH="${1:-$SCRIPT_DIR/../data/data.csv}"
STRATEGY="${2:-maxdiv}"
COV="${3:-ew}"

echo "=========================================="
echo "  MVO Portfolio C++ 运行"
echo "=========================================="
echo ""
echo "数据文件: $DATA_PATH"
echo "策略: $STRATEGY"
echo "协方差: $COV"
echo ""

# 生成输出文件名
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="$SCRIPT_DIR/output/$TIMESTAMP"
mkdir -p "$OUTPUT_DIR"

WEIGHT_FILE="$OUTPUT_DIR/w_${STRATEGY}_${COV}.csv"
RETURN_FILE="$OUTPUT_DIR/r_${STRATEGY}_${COV}.csv"

# 运行
"$BUILD_DIR/algo_cpp" "$DATA_PATH" \
    --strategy "$STRATEGY" \
    --cov "$COV" \
    --weights "$WEIGHT_FILE" \
    --output "$RETURN_FILE"

echo ""
echo "输出文件:"
echo "  权重: $WEIGHT_FILE"
echo "  收益: $RETURN_FILE"
