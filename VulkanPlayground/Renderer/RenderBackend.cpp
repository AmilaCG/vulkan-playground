#include "RenderBackend.h"

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/vec4.hpp>
#include <glm/mat4x4.hpp>

#include <iostream>
#include <cstring>

static constexpr int g_numDebugInstanceExtensions = 1;
static const char * g_debugInstanceExtensions[ g_numDebugInstanceExtensions ] = {
    VK_EXT_DEBUG_REPORT_EXTENSION_NAME
};

static constexpr int g_numDeviceExtensions = 1;
static const char * g_deviceExtensions[ g_numDeviceExtensions ] = {
    VK_KHR_SWAPCHAIN_EXTENSION_NAME
};

static constexpr int g_numValidationLayers = 1;
static const char * g_validationLayers[ g_numValidationLayers ] = {
    "VK_LAYER_KHRONOS_validation"
};

RenderBackend::RenderBackend() : m_window(nullptr), m_instance(nullptr), m_enableValidation(true)
{
}

RenderBackend::~RenderBackend() = default;

bool RenderBackend::WindowInit()
{
    glfwInit();

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    m_window = glfwCreateWindow(800, 600, "Vulkan window", nullptr, nullptr);

    uint32_t extensionCount = 0;
    vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, nullptr);

    std::cout << extensionCount << " extensions supported\n";

    return m_window != nullptr;
}

void RenderBackend::Init()
{
    if (!WindowInit())
    {
        throw std::runtime_error("Unable to initialize the GLFW window");
    }

    // Input and sound systems need to be tied to the new window
    SysInitInput();

    // Create the instance
    CreateInstance();

    // Create presentation surface
    CreateSurface();

    // Enumerate physical devices and get their properties
    SelectSuitablePhysicalDevice();

    // Create logical device and queues
    CreateLogicalDeviceAndQueues();

    // Create semaphores for image acquisition and rendering completion
    CreateSemaphores();

    // Create Query Pool
    CreateQueryPool();

    // Create Command Pool
    CreateCommandPool();

    // Create Command Buffer
    CreateCommandBuffer();

    // Setup the allocator
#if defined( ID_USE_AMD_ALLOCATOR )
    extern idCVar r_vkHostVisibleMemoryMB;
    extern idCVar r_vkDeviceLocalMemoryMB;

    VmaAllocatorCreateInfo createInfo = {};
    createInfo.physicalDevice = m_physicalDevice;
    createInfo.device = vkcontext.device;
    createInfo.preferredSmallHeapBlockSize = r_vkHostVisibleMemoryMB.GetInteger() * 1024 * 1024;
    createInfo.preferredLargeHeapBlockSize = r_vkDeviceLocalMemoryMB.GetInteger() * 1024 * 1024;

    vmaCreateAllocator( &createInfo, &vmaAllocator );
#else
    // vulkanAllocator.Init();
#endif

    // Start the Staging Manager
    // stagingManager.Init();

    // Create Swap Chain
    CreateSwapChain();

    // Create Render Targets
    CreateRenderTargets();

    // Create Render Pass
    CreateRenderPass();

    // Create Pipeline Cache
    CreatePipelineCache();

    // Create Frame Buffers
    CreateFrameBuffers();

    // Init RenderProg Manager
    // renderProgManager.Init();

    // Init Vertex Cache
    // vertexCache.Init( vkcontext.gpu.props.limits.minUniformBufferOffsetAlignment );
}

void RenderBackend::Shutdown()
{
    glfwDestroyWindow(m_window);
    glfwTerminate();
}

void RenderBackend::SysInitInput()
{

}

void RenderBackend::CreateInstance()
{
    VkApplicationInfo appInfo = {};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "Vulkan Playground";
    appInfo.applicationVersion = 1;
    appInfo.pEngineName = "Lumina";
    appInfo.engineVersion = 1;
    appInfo.apiVersion = VK_MAKE_VERSION(1, 0, 0);

    VkInstanceCreateInfo createInfo = {};
    createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    createInfo.pApplicationInfo = &appInfo;

    m_instanceExtensions.clear();
    m_deviceExtensions.clear();
    m_validationLayers.clear();

    for (const auto& extension : g_deviceExtensions)
    {
        m_deviceExtensions.emplace_back(extension);
    }

#ifdef NDEBUG
    m_enableValidation = false;
#else
    m_enableValidation = true;
#endif

    if (m_enableValidation)
    {
        for (const auto& extension : g_debugInstanceExtensions)
        {
            m_instanceExtensions.emplace_back(extension);
        }

        for (const auto& extension : g_validationLayers)
        {
            m_validationLayers.emplace_back(extension);
        }

        ValidateValidationLayers();
    }

    uint32_t glfwExtensionCount = 0;
    const char** glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

    createInfo.enabledExtensionCount = glfwExtensionCount;
    createInfo.ppEnabledExtensionNames = glfwExtensions;
    createInfo.enabledLayerCount = m_validationLayers.size();
    createInfo.ppEnabledLayerNames = m_validationLayers.data();

    if (vkCreateInstance(&createInfo, nullptr, &m_instance) != VK_SUCCESS)
    {
        throw std::runtime_error("Failed to create VK instance!");
    }
}

void RenderBackend::CreateSurface()
{
}

void RenderBackend::SelectSuitablePhysicalDevice()
{
}

void RenderBackend::CreateLogicalDeviceAndQueues()
{
}

void RenderBackend::CreateSemaphores()
{
}

void RenderBackend::CreateQueryPool()
{
}

void RenderBackend::CreateCommandPool()
{
}

void RenderBackend::CreateCommandBuffer()
{
}

void RenderBackend::CreateSwapChain()
{
}

void RenderBackend::CreateRenderTargets()
{
}

void RenderBackend::CreateRenderPass()
{
}

void RenderBackend::CreatePipelineCache()
{
}

void RenderBackend::CreateFrameBuffers()
{
}

void RenderBackend::ValidateValidationLayers()
{
    uint32 instanceLayerCount = 0;
    vkEnumerateInstanceLayerProperties(&instanceLayerCount, nullptr);

    std::vector<VkLayerProperties> instanceLayers;
    instanceLayers.resize(instanceLayerCount);
    vkEnumerateInstanceLayerProperties(&instanceLayerCount, instanceLayers.data());

    bool found = false;
    for (const auto& validationLayer : g_validationLayers)
    {
        for (const auto& instanceLayer : instanceLayers)
        {
            if (strcmp(validationLayer, instanceLayer.layerName) == 0)
            {
                found = true;
                break;
            }
        }
        if (!found)
        {
            printf("Cannot find validation layer: %s.\n", validationLayer);
        }
    }
}
