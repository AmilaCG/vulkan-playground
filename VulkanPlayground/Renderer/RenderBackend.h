#ifndef RENDERBACKEND_H
#define RENDERBACKEND_H

#include <vector>
#include <GLFW/glfw3.h>
#include <vulkan/vulkan_core.h>
#include "RenderCommon.h"

// Everything that is needed by the backend needs to be double buffered to allow it to run in
// parallel on a dual cpu machine
static constexpr uint32 NUM_FRAME_DATA = 2;

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

struct vulkanContext_t
{
    vertCacheHandle_t           jointCacheHandle;

    GPUInfo_t                   gpu;

    VkDevice                    device;
    int                         graphicsFamilyIdx;
    int                         presentFamilyIdx;
    VkQueue                     graphicsQueue;
    VkQueue                     presentQueue;

    VkFormat                    depthFormat;
    VkRenderPass                renderPass;
    VkRenderPass                renderPassResume;
    VkPipelineCache             pipelineCache;
    VkSampleCountFlagBits       sampleCount;
    bool                        supersampling;

    // idArray< idImage *, MAX_IMAGE_PARMS > imageParms;
};

class RenderBackend
{
public:
    RenderBackend();
    ~RenderBackend();

    void Init();
    void Shutdown();

private:
    // GLFW window init
    bool WindowInit();

    // Input and sound systems need to be tied to the new window
    void SysInitInput();

    // Create the instance
    void CreateInstance();

    // Create presentation surface
    void CreateSurface();

    // Enumerate physical devices and get their properties
    void SelectSuitablePhysicalDevice();

    // Create logical device and queues
    void CreateLogicalDeviceAndQueues();

    // Create semaphores for image acquisition and rendering completion
    void CreateSemaphores();

    // Create Query Pool
    void CreateQueryPool();

    // Create Command Pool
    void CreateCommandPool();

    // Create Command Buffer
    void CreateCommandBuffer();

    // Create Swap Chain
    void CreateSwapChain();

    // Create Render Targets
    void CreateRenderTargets();

    // Create Render Pass
    void CreateRenderPass();

    // Create Pipeline Cache
    void CreatePipelineCache();

    // Create Frame Buffers
    void CreateFrameBuffers();

    static void ValidateValidationLayers();

private:
    GLFWwindow*                     m_window;
    VkInstance                      m_instance;
    bool                            m_enableValidation;

    std::vector<const char*>        m_instanceExtensions;
    std::vector<const char*>        m_deviceExtensions;
    std::vector<const char*>        m_validationLayers;
};

#endif
