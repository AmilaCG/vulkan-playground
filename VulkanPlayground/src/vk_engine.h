#pragma once

#include "vk_descriptors.h"
#include "vk_types.h"

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
};

struct ComputePushConstants
{
    glm::vec4 data1;
    glm::vec4 data2;
    glm::vec4 data3;
    glm::vec4 data4;
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

private:
    void init_vulkan();
    void init_swapchain();
    void init_commands();
    void init_sync_structures();
    void create_swapchain(uint32_t width, uint32_t height);
    void destroy_swapchain();
    FrameData& get_current_frame();
    void draw_background(VkCommandBuffer cmd);
    void init_descriptors();
    void init_pipelines();
    void init_background_pipelines();
    void immediate_submit(std::function<void(VkCommandBuffer cmd)>&& function);
    void init_imgui();
    void draw_imgui(VkCommandBuffer cmd, VkImageView targetImageView);

    bool _isInitialized{false};
    int _frameNumber{0};
    bool _stopRendering{false};
    VkExtent2D _windowExtent{1700, 900};
    struct SDL_Window* _window{nullptr};

    VkInstance _instance{};
    VkDebugUtilsMessengerEXT _debug_messenger{}; // Vulkan debug output handle
    VkPhysicalDevice _chosenGPU{};
    VkDevice _device{};
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
    AllocatedImage _drawImage{};
    VkExtent2D _drawExtent{};

    DescriptorAllocator _globalDescriptorAllocator{};
    VkDescriptorSet _drawImageDescriptors{};
    VkDescriptorSetLayout _drawImageDescriptorLayout{};

    VkPipeline _gradientPipeline{};
    VkPipelineLayout _gradientPipelineLayout{};

    // Immediate submit structures
    VkFence _immFence{};
    VkCommandBuffer _immCommandBuffer{};
    VkCommandPool _immCommandPool{};
};
