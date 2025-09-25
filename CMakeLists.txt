cmake_minimum_required(VERSION 3.10)
project(MLPModelONNXRuntimeCPP)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(ONNXRUNTIME_DIR "/usr/local/onnxruntime")

include_directories(include)
include_directories(${ONNXRUNTIME_DIR}/include)
link_directories(${ONNXRUNTIME_DIR}/lib)

add_executable(mlp_main
    src/main.cpp
    src/MLPModel.cpp
)

target_link_libraries(mlp_main onnxruntime)
