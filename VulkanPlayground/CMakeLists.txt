cmake_minimum_required(VERSION 3.28)

file(TO_CMAKE_PATH $ENV{VCPKG_ROOT} vcpkg_path)
set(CMAKE_TOOLCHAIN_FILE "${vcpkg_path}/scripts/buildsystems/vcpkg.cmake")

project(VulkanPlayground)

set(CMAKE_CXX_STANDARD 20)

add_subdirectory(src)

find_package(Vulkan REQUIRED)
find_package(SDL2 CONFIG REQUIRED)
find_package(glm CONFIG REQUIRED)
find_package(Stb REQUIRED)
find_package(tinyobjloader CONFIG REQUIRED)
find_package(fastgltf CONFIG REQUIRED)
find_package(fmt CONFIG REQUIRED)
find_package(vk-bootstrap CONFIG REQUIRED)
find_package(VulkanMemoryAllocator CONFIG REQUIRED)
find_package(imgui CONFIG REQUIRED)
find_package(VulkanUtilityLibraries CONFIG REQUIRED)

target_link_libraries(${PROJECT_NAME} PRIVATE
        Vulkan::Vulkan
        SDL2::SDL2
        SDL2::SDL2main
        glm::glm-header-only
        tinyobjloader::tinyobjloader
        fastgltf::fastgltf
        fmt::fmt-header-only
        vk-bootstrap::vk-bootstrap
        GPUOpen::VulkanMemoryAllocator
        imgui::imgui
        Vulkan::UtilityHeaders
)

set(build_out_dir "$<TARGET_FILE_DIR:${PROJECT_NAME}>")
set(shader_src_dir "${CMAKE_SOURCE_DIR}/shaders")
set(shader_out_dir "${PROJECT_SOURCE_DIR}/shaders/compiled/")
## find all the shader files under the shaders folder
file(GLOB_RECURSE GLSL_SOURCE_FILES
        "${shader_src_dir}/*.frag"
        "${shader_src_dir}/*.vert"
        "${shader_src_dir}/*.comp"
)
if(WIN32)
    set(shader_compiler_exe "$ENV{VULKAN_SDK}/Bin/glslc.exe")
else()
    set(shader_compiler_exe "$ENV{VULKAN_SDK}/Bin/glslc")
endif()

## iterate each shader
foreach(GLSL ${GLSL_SOURCE_FILES})
    message(STATUS "BUILDING SHADER")
    get_filename_component(FILE_NAME ${GLSL} NAME)
    set(SPIRV "${PROJECT_SOURCE_DIR}/shaders/compiled/${FILE_NAME}.spv")
    message(STATUS ${GLSL})
    ##execute glslang command to compile that specific shader
    add_custom_command(
            OUTPUT ${SPIRV}
            COMMAND ${shader_compiler_exe} ${GLSL} -o "${SPIRV}"
            DEPENDS ${GLSL})
    list(APPEND SPIRV_BINARY_FILES ${SPIRV})
endforeach(GLSL)

add_custom_target(
        CompileShaders ALL
        DEPENDS ${SPIRV_BINARY_FILES}
)

add_dependencies(${PROJECT_NAME} CompileShaders)
