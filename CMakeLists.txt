cmake_minimum_required(VERSION 3.8)

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake;")
set(CMAKE_CONFIGURATION_TYPES Debug Release CACHE TYPE INTERNAL FORCE)

set(COTIRE_ADDITIONAL_PREFIX_HEADER_IGNORE_PATH "C:/Program Files (x86)/Windows Kits/10/Include")
include(cotire)

project(SS13RogueLike VERSION 1.0.0 LANGUAGES CXX)

# add thirdparty dependencies
add_subdirectory("${CMAKE_SOURCE_DIR}/third_party")

set(src
    "src/config.hpp"
    "src/program.cpp"
    "src/atmo_module.cpp"
    "src/atmo_module.hpp"
    "src/material.cpp"
    "src/material.hpp"
    "src/renderer.cpp"
    "src/renderer.hpp")

add_executable(SS13RogueLike ${src})
target_link_libraries(SS13RogueLike PRIVATE glfw vulkan libshaderc glm gsl)
target_compile_definitions(SS13RogueLike PRIVATE "DEBUG_VULKAN")
set_target_properties(SS13RogueLike PROPERTIES
    CXX_STANDARD 17
    OUTPUT_NAME_DEBUG "SS13RogueLike_d"
    OUTPUT_NAME_RELEASE "SS13RogueLike")
file(COPY "src/shaders" DESTINATION ${CMAKE_BINARY_DIR})

install(TARGETS SS13RogueLike
    RUNTIME DESTINATION "bin"
    LIBRARY DESTINATION "bin"
    ARCHIVE DESTINATION "bin")
install(DIRECTORY "src/shaders" DESTINATION "bin")

cotire(SS13RogueLike)
