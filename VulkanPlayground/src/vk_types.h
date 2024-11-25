// Include file for standard system include files, or project specific include files.
#pragma once

#include <memory>
#include <optional>
#include <string>
#include <vector>
#include <span>
#include <array>
#include <functional>
#include <stack>

#include <vulkan/vulkan.h>
#include <vulkan/vk_enum_string_helper.h>
#include <vk_mem_alloc.h>

#include <fmt/core.h>

#include <glm/mat4x4.hpp>
#include <glm/vec4.hpp>


#define VK_CHECK(x)                                                     \
    do {                                                                \
        VkResult err = x;                                               \
        if (err) {                                                      \
            fmt::println("Detected Vulkan error: {}", string_VkResult(err)); \
            abort();                                                    \
        }                                                               \
    } while (0)

struct DrawContext;

// Base class for a renderable dynamic object
class IRenderable
{
    virtual void prepare_draw(const glm::mat4& topMatrix, DrawContext& ctx) = 0;
};

struct Node : public IRenderable
{
    // Parent pointer must be a weak pointer to avoid circular dependencies
    std::weak_ptr<Node> parent;
    std::vector<std::shared_ptr<Node>> children;

    glm::mat4 localTransform;
    glm::mat4 worldTransform;

    void refresh_transform(const glm::mat4& parentMatrix)
    {
        worldTransform = parentMatrix * localTransform;
        for (auto child : children)
        {
            child->refresh_transform(worldTransform);
        }
    }

    virtual void prepare_draw(const glm::mat4& topMatrix, DrawContext& ctx)
    {
        for (auto& child : children)
        {
            child->prepare_draw(topMatrix, ctx);
        }
    }
};

enum class MaterialPass : uint8_t
{
    MainColor,
    Transparent,
    Other
};

struct MaterialPipeline
{
    VkPipeline pipeline;
    VkPipelineLayout layout;
};

struct MaterialInstance
{
    MaterialPipeline* pipeline;
    VkDescriptorSet materialSet;
    MaterialPass passType;
};

struct AllocatedImage
{
    VkImage image;
    VkImageView imageView;
    VmaAllocation allocation;
    VkExtent3D imageExtent;
    VkFormat imageFormat;
};

struct AllocatedBuffer
{
    VkBuffer buffer;
    VmaAllocation allocation;
    VmaAllocationInfo info;
};

// UV parameters are interleaved due to alignment limitations on GPUs. We want this
// structure to match the shader version so interleaving like this improves it.
struct Vertex
{
    glm::vec3 position;
    float uv_x;
    glm::vec3 normal;
    float uv_y;
    glm::vec4 color;
};

struct GPUMeshBuffers
{
    AllocatedBuffer indexBuffer;
    AllocatedBuffer vertexBuffer;
    VkDeviceAddress vertexBufferAddress;
};

struct GPUDrawPushConstants
{
    glm::mat4 worldMatrix;
    VkDeviceAddress vertexBuffer;
};
