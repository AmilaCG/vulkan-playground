#include "RenderBackend.h"

#include <iostream>
#include <cstring>

static constexpr int g_numDebugInstanceExtensions = 1;
static const char* g_debugInstanceExtensions[g_numDebugInstanceExtensions] = {
    VK_EXT_DEBUG_REPORT_EXTENSION_NAME
};

static constexpr int g_numDeviceExtensions = 1;
static const char* g_deviceExtensions[g_numDeviceExtensions] = {
    VK_KHR_SWAPCHAIN_EXTENSION_NAME
};

static constexpr int g_numValidationLayers = 1;
static const char* g_validationLayers[g_numValidationLayers] = {
    "VK_LAYER_KHRONOS_validation"
};

RenderBackend::RenderBackend() :
m_window(nullptr),
m_instance(),
m_surface(),
m_vkContext({}),
m_enableValidation(true)
{
}

RenderBackend::~RenderBackend() = default;

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
    // Destroy window surface
    vkDestroySurfaceKHR(m_instance, m_surface, nullptr);
    // Destroy the Instance
    vkDestroyInstance(m_instance, nullptr);

    glfwDestroyWindow(m_window);
    glfwTerminate();
}

bool RenderBackend::WindowInit()
{
    glfwInit();

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    m_window = glfwCreateWindow(1280, 720, "Vulkan Playground", nullptr, nullptr);

    return m_window != nullptr;
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
        throw std::runtime_error("Failed to create VK instance!\n");
    }
}

void RenderBackend::CreateSurface()
{
    VkWin32SurfaceCreateInfoKHR createInfo {};
    createInfo.sType = VK_STRUCTURE_TYPE_WIN32_SURFACE_CREATE_INFO_KHR;
    createInfo.hwnd = glfwGetWin32Window(m_window);
    createInfo.hinstance = GetModuleHandle(nullptr);

    if (vkCreateWin32SurfaceKHR(m_instance, &createInfo, nullptr, &m_surface) != VK_SUCCESS)
    {
        throw std::runtime_error("Failed to create window surface!\n");
    }
}

void RenderBackend::SelectSuitablePhysicalDevice()
{
    uint32_t numDevices = 0;
    vkEnumeratePhysicalDevices(m_instance, &numDevices, nullptr);
    if (numDevices == 0)
    {
        throw std::runtime_error("vkEnumeratePhysicalDevices returned zero devices.\n");
    }

    std::vector<VkPhysicalDevice> devices;
    devices.resize(numDevices);
    if (vkEnumeratePhysicalDevices(m_instance, &numDevices, devices.data()) != VK_SUCCESS)
    {
        throw std::runtime_error("Failed to fetch devices!\n");
    }

    std::vector<GPUInfo_t> gpus;
    gpus.resize(numDevices);

    for (uint32_t i = 0; i < numDevices; i++)
    {
        GPUInfo_t& gpu = gpus[i];
        gpu.device = devices[i];

        {
            // Get queues from the device
            uint32_t numQueues = 0;
            vkGetPhysicalDeviceQueueFamilyProperties(gpu.device, &numQueues, nullptr);
            if (numQueues == 0)
            {
                throw std::runtime_error("vkGetPhysicalDeviceQueueFamilyProperties returned zero queues.\n");
            }

            gpu.queueFamilyProps.resize(numQueues);
            vkGetPhysicalDeviceQueueFamilyProperties(gpu.device, &numQueues, gpu.queueFamilyProps.data());
            if (numQueues == 0)
            {
                throw std::runtime_error("vkGetPhysicalDeviceQueueFamilyProperties returned zero queues.\n");
            }
        }

        {
            // Get extensions supported by the device
            uint32_t numExtensions;
            vkEnumerateDeviceExtensionProperties(gpu.device, nullptr, &numExtensions, nullptr);
            if (numExtensions == 0)
            {
                throw std::runtime_error("vkEnumerateDeviceExtensionProperties returned zero extensions.\n");
            }

            gpu.extensionProps.resize(numExtensions);
            vkEnumerateDeviceExtensionProperties(gpu.device, nullptr, &numExtensions, gpu.extensionProps.data());
            if (numExtensions == 0)
            {
                throw std::runtime_error("vkEnumerateDeviceExtensionProperties returned zero extensions.\n");
            }
        }

        // Surface capabilities basically describes what kind of image you can render to the user
        vkGetPhysicalDeviceSurfaceCapabilitiesKHR(gpu.device, m_surface, &gpu.surfaceCaps);

        {
            // Get the supported surface formats. This includes image format and color space.
            uint32_t numFormats;
            vkGetPhysicalDeviceSurfaceFormatsKHR(gpu.device, m_surface, &numFormats, nullptr);
            if (numFormats == 0)
            {
                throw std::runtime_error("vkGetPhysicalDeviceSurfaceFormatsKHR returned zero formats.\n");
            }

            gpu.surfaceFormats.resize(numFormats);
            vkGetPhysicalDeviceSurfaceFormatsKHR(gpu.device, m_surface, &numFormats, gpu.surfaceFormats.data());
            if (numFormats == 0)
            {
                throw std::runtime_error("vkGetPhysicalDeviceSurfaceFormatsKHR returned zero formats.\n");
            }
        }

        {
            // Get supported presentation modes
            uint32_t numPresentModes;
            vkGetPhysicalDeviceSurfacePresentModesKHR(gpu.device, m_surface, &numPresentModes, nullptr);
            if (numPresentModes == 0)
            {
                throw std::runtime_error("vkGetPhysicalDeviceSurfacePresentModesKHR returned zero present modes.\n");
            }

            gpu.presentModes.resize(numPresentModes);
            vkGetPhysicalDeviceSurfacePresentModesKHR(gpu.device, m_surface, &numPresentModes, gpu.presentModes.data());
            if (numPresentModes == 0)
            {
                throw std::runtime_error("vkGetPhysicalDeviceSurfacePresentModesKHR returned zero present modes.\n");
            }
        }

        vkGetPhysicalDeviceMemoryProperties(gpu.device, &gpu.memProps);
        vkGetPhysicalDeviceProperties(gpu.device, &gpu.props);
        vkGetPhysicalDeviceFeatures(gpu.device, &gpu.features);
    }
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
    uint32_t instanceLayerCount = 0;
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
