#pragma once

#include "vk_descriptors.h"
#include "vk_types.h"
#include "vk_loader.h"

struct DeletionQueue
{
    std::stack<std::function<void()>> deletors;

    void push_function(std::function<void()>&& function)
    {
        deletors.push(function);
    }

    void flush()
    {
        while (!deletors.empty())
        {
            auto deletor = deletors.top();
            deletor();
            deletors.pop();
        }
    }
};

struct FrameData
{
    VkCommandPool _commandPool;
    VkCommandBuffer _mainCommandBuffer;

    VkSemaphore _swapchainSemaphore;
    VkSemaphore _renderSemaphore;
    VkFence _renderFence;

    DeletionQueue _deletionQueue;
    DescriptorAllocatorGrowable _frameDescriptors;
};

struct ComputePushConstants
{
    glm::vec4 data1;
    glm::vec4 data2;
    glm::vec4 data3;
    glm::vec4 data4;
};

struct ComputeEffect
{
    const char* name;

    VkPipeline pipeline;
    VkPipelineLayout layout;

    ComputePushConstants data;
};

struct GPUSceneData
{
    glm::mat4 view;
    glm::mat4 proj;
    glm::mat4 viewProj;
    glm::vec4 ambientColor;
    glm::vec4 sunlightDirection;
    glm::vec4 sunlightColor;
};

struct GLTFMetallicRoughness
{
    MaterialPipeline opaquePipeline;
    MaterialPipeline transparentPipeline;

    VkDescriptorSetLayout materialLayout;

    struct MaterialConstants
    {
        glm::vec4 colorFactors;
        glm::vec4 metalRoughFactors;
        // Padding, we need it for uniform buffers
        glm::vec4 extra[14];
    };

    struct MaterialResources
    {
        AllocatedImage colorImage;
        VkSampler colorSampler;
        AllocatedImage metalRoughImage;
        VkSampler metalRoughSampler;
        VkBuffer dataBuffer;
        uint32_t dataBufferOffset;
    };

    DescriptorWriter writer;

    void build_pipelines(VulkanEngine* engine);
    void clear_resources(VkDevice device);

    MaterialInstance write_material(VkDevice device,
                                    MaterialPass pass,
                                    const MaterialResources& resources,
                                    DescriptorAllocatorGrowable& descriptorAllocator);
};

struct MeshNode : public Node
{
    void draw(const glm::mat4& topMatrix, DrawContext& ctx) override;

    std::shared_ptr<MeshAsset> mesh;
};

struct RenderObject
{
    uint32_t indexCount;
    uint32_t firstIndex;
    VkBuffer indexBuffer;

    MaterialInstance* material;

    glm::mat4 transform;
    VkDeviceAddress vertexBufferAddress;
};

struct DrawContext
{
    std::vector<RenderObject> opaqueSurfaces;
};

constexpr unsigned int ONE_SEC_NS = 1000000000; // 1 second in nanoseconds
constexpr unsigned int FRAME_OVERLAP = 2;

class VulkanEngine
{
public:
    static VulkanEngine& Get();

    void init();
    void cleanup();
    void draw();
    void run();

    GPUMeshBuffers upload_mesh(std::span<uint32_t> indices, std::span<Vertex> vertices);

    VkDevice _device{};
    VkDescriptorSetLayout _gpuSceneDataDescriptorLayout{};
    AllocatedImage _drawImage{};
    AllocatedImage _depthImage{};

private:
    void init_vulkan();
    void init_swapchain();
    void init_commands();
    void init_sync_structures();
    void create_swapchain(uint32_t width, uint32_t height);
    void destroy_swapchain();
    FrameData& get_current_frame();
    void draw_background(VkCommandBuffer cmd);
    void draw_geometry(VkCommandBuffer cmd);
    void init_descriptors();
    void init_pipelines();
    void init_background_pipelines();
    void immediate_submit(std::function<void(VkCommandBuffer cmd)>&& function);
    void init_imgui();
    void draw_imgui(VkCommandBuffer cmd, VkImageView targetImageView);
    AllocatedBuffer create_buffer(size_t allocSize, VkBufferUsageFlags usage, VmaMemoryUsage memoryUsage);
    void destroy_buffer(const AllocatedBuffer& buffer);
    void init_mesh_pipeline();
    void init_default_data();
    void resize_swapchain();
    AllocatedImage create_image(VkExtent3D size, VkFormat format, VkImageUsageFlags usage, bool mipmapped = false);
    AllocatedImage create_image(void* data, VkExtent3D size, VkFormat format, VkImageUsageFlags usage, bool mipmapped = false);
    void destroy_image(const AllocatedImage& image);
    void update_scene();

    bool _isInitialized{false};
    int _frameNumber{0};
    bool _stopRendering{false};
    VkExtent2D _windowExtent{1700, 900};
    struct SDL_Window* _window{nullptr};

    VkInstance _instance{};
    VkDebugUtilsMessengerEXT _debug_messenger{}; // Vulkan debug output handle
    VkPhysicalDevice _chosenGPU{};
    VkSurfaceKHR _surface{};

    VkSwapchainKHR _swapchain{};
    VkFormat _swapchainImageFormat{};

    std::vector<VkImage> _swapchainImages;
    std::vector<VkImageView> _swapchainImageViews;
    VkExtent2D _swapchainExtent{};

    FrameData _frames[FRAME_OVERLAP]{};

    VkQueue _graphicsQueue{};
    uint32_t _graphicsQueueFamily{};

    DeletionQueue _mainDeletionQueue;

    VmaAllocator _allocator{};

    // Draw resources
    VkExtent2D _drawExtent{};
    float _renderScale = 1.0f;

    AllocatedImage _whiteImage{};
    AllocatedImage _blackImage{};
    AllocatedImage _greyImage{};
    AllocatedImage _errorCheckboardImage{};

    VkSampler _defaultSamplerLinear{};
    VkSampler _defaultSamplerNearest{};

    DescriptorAllocatorGrowable _globalDescriptorAllocator{};
    VkDescriptorSet _drawImageDescriptors{};
    VkDescriptorSetLayout _drawImageDescriptorLayout{};
    VkDescriptorSetLayout _singleImageDescriptorLayout{};

    // Immediate submit structures
    VkFence _immFence{};
    VkCommandBuffer _immCommandBuffer{};
    VkCommandPool _immCommandPool{};

    VkPipelineLayout _gradientPipelineLayout{};
    std::vector<ComputeEffect> backgroundEffects;
    int currentBackgroundEffect{0};

    VkPipelineLayout _meshPipelineLayout{};
    VkPipeline _meshPipeline{};

    std::vector<std::shared_ptr<MeshAsset>> _testMeshes;

    GPUSceneData _sceneData{};

    MaterialInstance _defaultMaterialData{};
    GLTFMetallicRoughness _metalRoughMaterial{};

    DrawContext _mainDrawContext{};
    std::unordered_map<std::string, std::shared_ptr<Node>> _loadedNodes;
};
