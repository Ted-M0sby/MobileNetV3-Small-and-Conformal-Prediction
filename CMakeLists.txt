cmake_minimum_required(VERSION 3.16)
project(gongxingyuce_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(gongxingyuce
    src/main.cpp
)

target_include_directories(gongxingyuce PRIVATE include)

if(MSVC)
    target_compile_options(gongxingyuce PRIVATE /W4)
else()
    target_compile_options(gongxingyuce PRIVATE -Wall -Wextra -Wpedantic)
endif()
