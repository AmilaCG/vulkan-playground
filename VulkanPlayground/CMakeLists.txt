cmake_minimum_required(VERSION 3.28)

file(TO_CMAKE_PATH $ENV{VCPKG_ROOT} vcpkg_path)
set(CMAKE_TOOLCHAIN_FILE "${vcpkg_path}/scripts/buildsystems/vcpkg.cmake")

project(VulkanPlayground)

set(CMAKE_CXX_STANDARD 20)

add_executable(VulkanPlayground
        main.cpp
        Renderer/RenderBackend.cpp)

# Uncomment following definition to disable validation layers
#add_definitions(-DVALIDATION_OFF)

find_package(Vulkan REQUIRED)
find_package(glfw3 CONFIG REQUIRED)
find_package(glm CONFIG REQUIRED)
find_package(Stb REQUIRED)

target_link_libraries(VulkanPlayground PRIVATE
        Vulkan::Vulkan
        glfw
        glm::glm-header-only)

if(WIN32)
    add_custom_target(CompileShaders ALL
            COMMAND cmd /c $ENV{VULKAN_SDK}/Bin/glslc.exe ${CMAKE_SOURCE_DIR}/shaders/shader.vert -o  vert.spv
            COMMAND cmd /c $ENV{VULKAN_SDK}/Bin/glslc.exe ${CMAKE_SOURCE_DIR}/shaders/shader.frag -o  frag.spv
    )
    add_dependencies(VulkanPlayground CompileShaders)
endif()
