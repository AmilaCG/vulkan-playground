#ifndef RENDERBACKEND_H
#define RENDERBACKEND_H

#define VK_USE_PLATFORM_WIN32_KHR
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#define GLFW_EXPOSE_NATIVE_WIN32
#include <GLFW/glfw3native.h>

#include <vulkan/vulkan_core.h>
#include <vector>

// Everything that is needed by the backend needs to be double buffered to allow it to run in
// parallel on a dual cpu machine
static constexpr uint32_t NUM_FRAME_DATA = 2;

struct GPUInfo_t
{
    VkPhysicalDevice                     device;
    VkPhysicalDeviceProperties           props;
    VkPhysicalDeviceMemoryProperties     memProps;
    VkPhysicalDeviceFeatures             features;
    VkSurfaceCapabilitiesKHR             surfaceCaps;
    std::vector<VkSurfaceFormatKHR>      surfaceFormats;
    std::vector<VkPresentModeKHR>        presentModes;
    std::vector<VkQueueFamilyProperties> queueFamilyProps;
    std::vector<VkExtensionProperties>   extensionProps;
};

struct VulkanContext_t
{
    GPUInfo_t                   gpu;

    VkDevice                    device;
    int                         graphicsFamilyIdx;
    int                         presentFamilyIdx;
    VkQueue                     graphicsQueue;
    VkQueue                     presentQueue;

    VkFormat                    depthFormat;
    VkRenderPass                renderPass;
    VkPipelineCache             pipelineCache;
    VkSampleCountFlagBits       sampleCount;
    bool                        supersampling;
};

class RenderBackend
{
public:
    RenderBackend();
    ~RenderBackend();

    void Init();
    void Shutdown();
    void RunRenderLoop();

private:
    bool WindowInit();
    void SysInitInput();
    void CreateInstance();
    void CreateSurface();
    void SelectSuitablePhysicalDevice();
    void CreateLogicalDeviceAndQueues();
    void CreateSemaphores();
    void CreateQueryPool();
    void CreateCommandPool();
    void CreateCommandBuffers();
    void CreateSwapChain();
    void CreateRenderTargets();
    void CreateRenderPass();
    void CreatePipelineCache();
    void CreateGraphicsPipeline();
    void CreateFrameBuffers();
    void RecordCommandbuffer(const VkCommandBuffer& commandBuffer, uint32_t imageIndex);
    void DrawFrame();
    void RecreateSwapchain();

    VkExtent2D ChooseSurfaceExtent(const VkSurfaceCapabilitiesKHR& caps);
    VkShaderModule CreateShaderModule(const std::vector<char>& code);

private:
    GLFWwindow*                     m_window;
    VkInstance                      m_instance{};
    VkSurfaceKHR                    m_surface{};
    VkPresentModeKHR                m_presentMode{};
    VulkanContext_t                 m_vkCtx{};
    VkPhysicalDevice                m_physicalDevice{};
    bool                            m_enableValidation;

    uint32_t                        m_currentFrame;

    std::vector<const char*>        m_instanceExtensions;
    std::vector<const char*>        m_deviceExtensions;
    std::vector<const char*>        m_validationLayers;

    std::vector<VkSemaphore>        m_acquireSemaphores;
    std::vector<VkSemaphore>        m_renderCompleteSemaphores;
    std::vector<VkFence>            m_commandBufferFences;

    VkCommandPool                   m_commandPool{};
    std::vector<VkCommandBuffer>    m_commandBuffers;
    VkSwapchainKHR                  m_swapchain{};
    VkFormat                        m_swapchainFormat{};
    VkExtent2D                      m_swapchainExtent{};
    std::vector<VkImage>            m_swapchainImages;
    std::vector<VkImageView>        m_swapchainViews;
    std::vector<VkFramebuffer>      m_swapchainFramebuffers;

    VkPipelineLayout                m_pipelineLayout{};
    VkPipeline                      m_pipeline{};
};

#endif
