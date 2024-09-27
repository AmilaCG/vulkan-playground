#include "vk_loader.h"

#include <iostream>
#include <stb_image.h>
#include <glm/gtx/quaternion.hpp>
#include <fastgltf/glm_element_traits.hpp>
#include <fastgltf/core.hpp>
#include <fastgltf/tools.hpp>

#include "vk_loader.h"
#include "vk_engine.h"
#include "vk_initializers.h"
#include "vk_types.h"

std::optional<std::vector<std::shared_ptr<MeshAsset>>> load_gltf_meshes(VulkanEngine* engine,
                                                                        std::filesystem::path filePath)
{
    std::cout << "Loading GLTF: " << filePath << std::endl;

    fastgltf::GltfDataBuffer data;
    data.loadFromFile(filePath);

    constexpr auto gltfOptions = fastgltf::Options::LoadGLBBuffers |
                                 fastgltf::Options::LoadExternalBuffers;

    fastgltf::Asset gltf;
    fastgltf::Parser parser{};

    auto load = parser.loadGltf(&data, filePath.parent_path(), gltfOptions);
    if (load)
    {
        gltf = std::move(load.get());
    }
    else
    {
        fmt::print("Failed to load glTF: {} \n", fastgltf::to_underlying(load.error()));
        return {};
    }
}
