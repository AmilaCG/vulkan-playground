cmake_minimum_required(VERSION 3.28)

file(TO_CMAKE_PATH $ENV{VCPKG_ROOT} vcpkg_path)
set(CMAKE_TOOLCHAIN_FILE "${vcpkg_path}/scripts/buildsystems/vcpkg.cmake")

project(VulkanPlayground)

set(CMAKE_CXX_STANDARD 20)

add_executable(${PROJECT_NAME}
        main.cpp
        Renderer/RenderBackend.cpp)

# Uncomment following definition to disable validation layers
#add_definitions(-DVALIDATION_OFF)

find_package(Vulkan REQUIRED)
find_package(glfw3 CONFIG REQUIRED)
find_package(glm CONFIG REQUIRED)
find_package(Stb REQUIRED)
find_package(tinyobjloader CONFIG REQUIRED)

target_link_libraries(${PROJECT_NAME} PRIVATE
        Vulkan::Vulkan
        glfw
        glm::glm-header-only
        tinyobjloader::tinyobjloader)

set(build_out_dir "$<TARGET_FILE_DIR:${PROJECT_NAME}>")
set(shader_src_dir "${CMAKE_SOURCE_DIR}/shaders")
if(WIN32)
    set(shader_compiler_exe "$ENV{VULKAN_SDK}/Bin/glslc.exe")

    add_custom_target(CompileShaders ALL
            COMMAND ${CMAKE_COMMAND} -E make_directory ${build_out_dir}
            COMMAND cmd /c ${shader_compiler_exe} ${shader_src_dir}/shader.vert -o ${build_out_dir}/vert.spv
            COMMAND cmd /c ${shader_compiler_exe} ${shader_src_dir}/shader.frag -o ${build_out_dir}/frag.spv
    )

    add_dependencies(${PROJECT_NAME} CompileShaders)
endif()

add_custom_target(ClearAssets ALL
        COMMAND ${CMAKE_COMMAND} -E rm -rf
        ${build_out_dir}/Assets/)

add_custom_target(CopyAssets ALL
        COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_SOURCE_DIR}/Assets/ ${build_out_dir}/Assets/)

add_dependencies(${PROJECT_NAME} CopyAssets)
add_dependencies(CopyAssets ClearAssets)
