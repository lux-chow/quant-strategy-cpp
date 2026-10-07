cmake_minimum_required(VERSION 3.15)
project(MVOPortfolio VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Release 模式优化
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Release")
    add_compile_options(-O3 -march=native -DNDEBUG)
    add_link_options(-O3 -march=native)
endif()

# 警告处理
add_compile_options(-Wall -Wextra -Wpedantic)

# Eigen3 - 使用本地 thirdparty
set(EIGEN3_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/thirdparty/eigen/eigen-3.4.0")
include_directories(${EIGEN3_INCLUDE_DIR})

# NLopt - 使用本地构建
set(NLOPT_BUILD_DIR "${CMAKE_SOURCE_DIR}/thirdparty/nlopt/build")
set(NLOPT_SOURCE_API_DIR "${CMAKE_SOURCE_DIR}/thirdparty/nlopt/src/api")
set(NLOPT_LIBRARY "${NLOPT_BUILD_DIR}/libnlopt.so")

if(EXISTS ${NLOPT_LIBRARY})
    message(STATUS "Found NLopt (local build): ${NLOPT_LIBRARY}")
else()
    message(FATAL_ERROR "NLopt not found at ${NLOPT_LIBRARY}")
endif()

include_directories(${NLOPT_BUILD_DIR} ${NLOPT_SOURCE_API_DIR})

# 静态库：mvo_core
set(MVO_CORE_SOURCES
    src/data_loader.cpp
    src/preprocess.cpp
    src/covariance.cpp
    src/objective.cpp
    src/constraints.cpp
    src/optimizer.cpp
    src/metrics.cpp
    src/mvo.cpp
)

add_library(mvo_core STATIC ${MVO_CORE_SOURCES})
target_include_directories(mvo_core PUBLIC 
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(mvo_core PUBLIC ${NLOPT_LIBRARY})

# 可执行文件：algo_cpp
add_executable(algo_cpp src/main.cpp)
target_link_libraries(algo_cpp PRIVATE mvo_core ${NLOPT_LIBRARY})

# 可执行文件：algo_cpp_verify
add_executable(algo_cpp_verify src/main_verify.cpp)
target_link_libraries(algo_cpp_verify PRIVATE mvo_core ${NLOPT_LIBRARY})

# 安装目标
install(TARGETS mvo_core algo_cpp algo_cpp_verify
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

install(DIRECTORY include/ DESTINATION include)

# 测试
enable_testing()
add_test(NAME mvo_core_test COMMAND algo_cpp_verify)
