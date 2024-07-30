cmake_minimum_required(VERSION 3.28)

file(TO_CMAKE_PATH $ENV{VCPKG_ROOT} vcpkg_path)
set(CMAKE_TOOLCHAIN_FILE "${vcpkg_path}/scripts/buildsystems/vcpkg.cmake")

project(VulkanPlayground)

set(CMAKE_CXX_STANDARD 20)

add_executable(VulkanPlayground
        main.cpp
        Renderer/RenderBackend.cpp)

find_package(Vulkan REQUIRED)
find_package(glfw3 CONFIG REQUIRED)
find_package(glm CONFIG REQUIRED)

target_link_libraries(VulkanPlayground PRIVATE
        Vulkan::Vulkan
        glfw
        glm::glm-header-only)
