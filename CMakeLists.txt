cmake_minimum_required(VERSION 3.5)

project(Yolov8CPPInference VERSION 0.1)

set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_CXX_STANDARD 17)

INCLUDE_DIRECTORIES(["/PATH/TO/TENSORFLOW/ROOT"])
INCLUDE_DIRECTORIES(["/PATH/TO/flatbuffers/include"])

ADD_LIBRARY(tensorflowlite SHARED IMPORTED)
set_property(TARGET tensorflowlite PROPERTY IMPORTED_LOCATION [/PATH/TO/LIB])

# OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
# !OpenCV

include_directories(src)

set(PROJECT_SOURCES
    main.cpp
    src/yolov8.h
    src/yolov8.cpp
    src/utils.h
    src/utils.cpp
)

add_executable(Yolov8CPPInference ${PROJECT_SOURCES})
target_link_libraries(Yolov8CPPInference tensorflowlite ${OpenCV_LIBS})
