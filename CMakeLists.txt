cmake_minimum_required(VERSION 3.8)

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
set(CMAKE_CONFIGURATION_TYPES Debug Release CACHE TYPE INTERNAL FORCE )
set(COTIRE_ADDITIONAL_PREFIX_HEADER_IGNORE_PATH "C:/Program Files (x86)/Windows Kits/10/Include")

include(cotire)

project (SS13RogueLike VERSION 1.0.0 LANGUAGES CXX)
add_definitions(-DWIN64)

include_directories(
    "third_party/"
    "third_party/glfw-3.2.1.bin.WIN64/include"
)

link_directories(
    "$ENV{VULKAN_SDK}/Lib"
    "third_party/glfw-3.2.1.bin.WIN64/lib-vc2015"
)

set(src
    "src/program.cpp"
    "src/atmo_module.cpp"
    "src/atmo_module.hpp"
    "src/renderer.cpp"
    "src/renderer.hpp"
)

add_executable(SS13RogueLike ${src})

target_link_libraries(SS13RogueLike
    PUBLIC vulkan-1.lib glfw3.lib
)

set_target_properties(SS13RogueLike
    PROPERTIES
    OUTPUT_NAME_DEBUG "SS13RogueLike_d"
    OUTPUT_NAME_RELEASE "SS13RogueLike"
    RUNTIME_OUTPUT_DIRECTORY_DEBUG "${CMAKE_SOURCE_DIR}/bin"
    RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_SOURCE_DIR}/bin"
)

cotire(SS13RogueLike)
