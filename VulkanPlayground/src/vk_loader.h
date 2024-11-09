#pragma once

#include <filesystem>

#include "vk_descriptors.h"
#include "vk_types.h"

struct GLTFMaterial
{
    MaterialInstance data;
};

struct GeoSurface
{
    uint32_t startIndex;
    uint32_t count;
    std::shared_ptr<GLTFMaterial> material;
};

struct MeshAsset
{
    std::string name;

    std::vector<GeoSurface> surfaces;
    GPUMeshBuffers meshBuffers;
};

class VulkanEngine;

struct LoadedGLTF : public IRenderable
{
    ~LoadedGLTF() { clearAll(); };

    virtual void draw(const glm::mat4& topMatrix, DrawContext& ctx);

private:
    void clearAll();

public:
    // Storage for all the data on a given glTF file
    std::unordered_map<std::string, std::shared_ptr<MeshAsset>> meshes;
    std::unordered_map<std::string, std::shared_ptr<Node>> nodes;
    std::unordered_map<std::string, AllocatedImage> images;
    std::unordered_map<std::string, std::shared_ptr<GLTFMaterial>> materials;

    // Nodes that don't have a parent, for iterating through the file in tree order
    std::vector<std::shared_ptr<Node>> topNodes;

    std::vector<VkSampler> samplers;
    DescriptorAllocatorGrowable descriptorPool{};
    AllocatedBuffer materialDataBuffer;
    VulkanEngine* creator;
};

std::optional<std::shared_ptr<LoadedGLTF>> load_gltf(VulkanEngine* engine, std::string_view filePath);

std::optional<std::vector<std::shared_ptr<MeshAsset>>> load_gltf_meshes(VulkanEngine* engine,
                                                                        std::filesystem::path filePath);
