add_library(glfw SHARED IMPORTED GLOBAL)
set(GLFW_DLL "${CMAKE_CURRENT_SOURCE_DIR}/glfw/lib-vc2015/glfw3.dll")
set_target_properties(glfw PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_CURRENT_SOURCE_DIR}/glfw/include"
    IMPORTED_IMPLIB "${CMAKE_CURRENT_SOURCE_DIR}/glfw/lib-vc2015/glfw3dll.lib"
    IMPORTED_LOCATION "${GLFW_DLL}")
file(COPY ${GLFW_DLL} DESTINATION ${CMAKE_BINARY_DIR})
install(FILES ${GLFW_DLL} DESTINATION "bin")

file(TO_CMAKE_PATH $ENV{VULKAN_SDK} VulkanPath)
add_library(vulkan STATIC IMPORTED GLOBAL)
set_target_properties(vulkan PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${VulkanPath}/Include"
    IMPORTED_LOCATION "${VulkanPath}/Lib/vulkan-1.lib")

add_library(glm INTERFACE IMPORTED GLOBAL)
set_target_properties(glm PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_CURRENT_SOURCE_DIR}")

add_library(gsl INTERFACE IMPORTED GLOBAL)
set_target_properties(gsl PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_CURRENT_SOURCE_DIR}")

set(SHADERC_SKIP_INSTALL ON)
set(SHADERC_SKIP_TESTS ON)
set(SHADERC_ENABLE_SHARED_CRT ON)
file(TO_CMAKE_PATH $ENV{VULKAN_SDK} VulkanPath)
add_subdirectory("${VulkanPath}/shaderc/" shaderc EXCLUDE_FROM_ALL)

add_library(libshaderc INTERFACE IMPORTED GLOBAL)
set_target_properties(libshaderc PROPERTIES
    INTERFACE_LINK_LIBRARIES shaderc)
