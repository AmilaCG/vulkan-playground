#include "RenderBackend.h"

#include <iostream>
#include <fstream>
#include <cstring>
#include <unordered_set>

constexpr uint32_t SCR_WIDTH = 1280;
constexpr uint32_t SCR_HEIGHT = 720;

constexpr std::string VERT_SHADER_PATH = "vert.spv";
constexpr std::string FRAG_SHADER_PATH = "frag.spv";

static const char* g_debugInstanceExtensions[] = {
    VK_EXT_DEBUG_REPORT_EXTENSION_NAME
};

static const char* g_deviceExtensions[] = {
    VK_KHR_SWAPCHAIN_EXTENSION_NAME
};

static const char* g_validationLayers[] = {
    "VK_LAYER_KHRONOS_validation", "VK_LAYER_LUNARG_monitor"
};

static void ValidateValidationLayers()
{
    uint32_t instanceLayerCount = 0;
    vkEnumerateInstanceLayerProperties(&instanceLayerCount, nullptr);

    std::vector<VkLayerProperties> instanceLayers(instanceLayerCount);
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

static bool CheckPhysicalDeviceExtensionSupport(GPUInfo_t& gpu, std::vector<const char*>& requiredExt)
{
    const int required = requiredExt.size();
    int available = 0;

    for (const char* requiredExtension : requiredExt)
    {
        for (const VkExtensionProperties& extensionProp : gpu.extensionProps)
        {
            if (strcmp(requiredExtension, extensionProp.extensionName) == 0)
            {
                available++;
                break;
            }
        }
    }

    return available == required;
}

static VkSurfaceFormatKHR ChooseSurfaceFormat(std::vector<VkSurfaceFormatKHR>& formats)
{
    // If Vulkan returned an unknown format, then just force what we want
    if (formats.size() == 1 && formats[0].format == VK_FORMAT_UNDEFINED)
    {
        VkSurfaceFormatKHR result;
        result.format = VK_FORMAT_B8G8R8A8_UNORM;
        result.colorSpace = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;
        return result;
    }

    // Favor 32 bit rgba and srgb nonlinear colorspace
    for (const VkSurfaceFormatKHR& fmt : formats)
    {
        if (fmt.format == VK_FORMAT_B8G8R8A8_UNORM && fmt.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR)
        {
            return fmt;
        }
    }

    // If all else fails, just return what's available
    return formats[0];
}

static VkPresentModeKHR ChoosePresentMode(std::vector<VkPresentModeKHR>& modes)
{
    // VK_PRESENT_MODE_FIFO_KHR    - Cap FPS at screen refresh rate
    // VK_PRESENT_MODE_MAILBOX_KHR - No FPS cap
    constexpr VkPresentModeKHR desiredMode = VK_PRESENT_MODE_MAILBOX_KHR;

    for (const VkPresentModeKHR& mode : modes)
    {
        if (mode == desiredMode)
        {
            return desiredMode;
        }
    }

    // If we couldn't find desired mode, then default to FIFO which is always available
    return VK_PRESENT_MODE_FIFO_KHR;
}

static VkFormat ChooseSupportedFormat(
    const VkPhysicalDevice& physicalDevice,
    VkFormat* formats,
    int numFormats,
    const VkImageTiling& tiling,
    const VkFormatFeatureFlags& features)
{
    for ( int i = 0; i < numFormats; ++i )
    {
        const VkFormat& format = formats[i];
        VkFormatProperties props;
        vkGetPhysicalDeviceFormatProperties(physicalDevice, format, &props);

        if (tiling == VK_IMAGE_TILING_LINEAR && (props.linearTilingFeatures & features) == features)
        {
            return format;
        }

        if (tiling == VK_IMAGE_TILING_OPTIMAL && (props.optimalTilingFeatures & features) == features)
        {
            return format;
        }
    }

    throw std::runtime_error("Failed to find a supported format.\n");
    return VK_FORMAT_UNDEFINED;
}

static void ReadShaderFile(const std::string& filename, std::vector<char>& buffer)
{
    std::ifstream file(filename, std::ios::ate | std::ios::binary);
    if (!file.is_open())
    {
        std::cout << "Failed to open " << filename << std::endl;
        throw std::runtime_error("Failed to open file");
    }

    // The above flag (std::ios::ate) will start reading the file from the end. The benifit of
    // reading from the end is that we can use the read position to determine the file size.
    size_t fileSize = static_cast<size_t>(file.tellg());

    if (!buffer.empty()) { buffer.clear(); }
    buffer.resize(fileSize);

    file.seekg(0);
    file.read(buffer.data(), fileSize);
    file.close();
}

RenderBackend::RenderBackend() : m_window(nullptr),
                                 m_enableValidation(true),
                                 m_currentFrame(0),
                                 m_acquireSemaphores(NUM_FRAME_DATA),
                                 m_renderCompleteSemaphores(NUM_FRAME_DATA),
                                 m_commandBufferFences(NUM_FRAME_DATA),
                                 m_commandBuffers(NUM_FRAME_DATA),
                                 m_swapchainImages(NUM_FRAME_DATA),
                                 m_swapchainViews(NUM_FRAME_DATA),
                                 m_swapchainFramebuffers(NUM_FRAME_DATA)
{
#ifdef NDEBUG
    m_enableValidation = false;
#else
    m_enableValidation = true;
#endif
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
    CreateCommandBuffers();

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

    CreateGraphicsPipeline();

    // Create Frame Buffers
    CreateFrameBuffers();

    // Init RenderProg Manager
    // renderProgManager.Init();

    // Init Vertex Cache
    // vertexCache.Init( vkcontext.gpu.props.limits.minUniformBufferOffsetAlignment );
}

void RenderBackend::Shutdown()
{
    for (const auto& framebuffer : m_swapchainFramebuffers)
    {
        vkDestroyFramebuffer(m_vkCtx.device, framebuffer, nullptr);
    }

    vkDestroyPipeline(m_vkCtx.device, m_pipeline, nullptr);
    vkDestroyPipelineLayout(m_vkCtx.device, m_pipelineLayout, nullptr);
    vkDestroyRenderPass(m_vkCtx.device, m_vkCtx.renderPass, nullptr);

    for (const auto& imageView : m_swapchainViews)
    {
        vkDestroyImageView(m_vkCtx.device, imageView, nullptr);
    }
    vkDestroySwapchainKHR(m_vkCtx.device, m_swapchain, nullptr);

    // Destroy fences
    for (const VkFence& fence : m_commandBufferFences)
    {
        vkDestroyFence(m_vkCtx.device, fence, nullptr);
    }

    // TODO: Is it necessary to release command buffers? Validation layer doesn't complain
    vkDestroyCommandPool(m_vkCtx.device, m_commandPool, nullptr);

    // Destroy semaphores
    for (const VkSemaphore& semaphore : m_acquireSemaphores)
    {
        vkDestroySemaphore(m_vkCtx.device, semaphore, nullptr);
    }
    for (const VkSemaphore& semaphore : m_renderCompleteSemaphores)
    {
        vkDestroySemaphore(m_vkCtx.device, semaphore, nullptr);
    }

    // Destroy logical device
    vkDestroyDevice(m_vkCtx.device, nullptr);
    // Destroy window surface
    vkDestroySurfaceKHR(m_instance, m_surface, nullptr);
    vkDestroyInstance(m_instance, nullptr);

    glfwDestroyWindow(m_window);
    glfwTerminate();
}

void RenderBackend::RunRenderLoop()
{
    while (!glfwWindowShouldClose(m_window))
    {
        glfwPollEvents();
        DrawFrame();
    }

    // Wait for the logical device to finish operations before exiting
    vkDeviceWaitIdle(m_vkCtx.device);
}

bool RenderBackend::WindowInit()
{
    glfwInit();

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    m_window = glfwCreateWindow(SCR_WIDTH, SCR_HEIGHT, "Vulkan Playground", nullptr, nullptr);

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
    appInfo.pEngineName = "Playground";
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

    std::vector<VkPhysicalDevice> devices(numDevices);
    if (vkEnumeratePhysicalDevices(m_instance, &numDevices, devices.data()) != VK_SUCCESS)
    {
        throw std::runtime_error("Failed to fetch devices!\n");
    }

    std::vector<GPUInfo_t> gpus(numDevices);

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

    for (GPUInfo_t& gpu : gpus)
    {
        int graphicsIdx = -1;
        int presentIdx = -1;

        // Remember when we created our instance we got all those device extensions?
        // Now we need to make sure our physical device supports them.
        if (!CheckPhysicalDeviceExtensionSupport(gpu, m_deviceExtensions))
        {
            continue;
        }

        if (gpu.surfaceFormats.empty())
        {
            continue;
        }

        if (gpu.presentModes.empty())
        {
            continue;
        }

        // Loop through the queue family properties looking for both a graphics
        // and a present queue.

        // Find graphics queue family
        for (int i = 0; i < gpu.queueFamilyProps.size(); i++)
        {
            VkQueueFamilyProperties& props = gpu.queueFamilyProps[i];

            if (props.queueCount == 0)
            {
                continue;
            }

            if (props.queueFlags & VK_QUEUE_GRAPHICS_BIT)
            {
                graphicsIdx = i;
                break;
            }
        }

        // Find present queue family
        for (int i = 0; i < gpu.queueFamilyProps.size(); i++)
        {
            VkQueueFamilyProperties& props = gpu.queueFamilyProps[i];

            if (props.queueCount == 0)
            {
                continue;
            }

            VkBool32 supportPresentation = VK_FALSE;
            vkGetPhysicalDeviceSurfaceSupportKHR(gpu.device, i, m_surface, &supportPresentation);
            if (supportPresentation)
            {
                presentIdx = i;
                break;
            }
        }

        // Did we find a device supporting both graphics and present
        if (graphicsIdx >= 0 && presentIdx >= 0)
        {
            m_vkCtx.graphicsFamilyIdx = graphicsIdx;
            m_vkCtx.presentFamilyIdx = presentIdx;
            m_physicalDevice = gpu.device;
            m_vkCtx.gpu = gpu;
            std::cout << "Selected device: " << gpu.props.deviceName << std::endl;

            return;
        }
    }
}

void RenderBackend::CreateLogicalDeviceAndQueues()
{
    // Logical device is simply an interface to the physical device. It exposes the
    // underlying API of interacting with the device.

    // Add each family index to a list. Don't do duplicates.
    std::unordered_set<int> uniqueIdx;
    uniqueIdx.insert(m_vkCtx.graphicsFamilyIdx);
    uniqueIdx.insert(m_vkCtx.presentFamilyIdx);

    std::vector<VkDeviceQueueCreateInfo> devQInfo;

    const float priority = 1.0f;
    for (const int idx : uniqueIdx)
    {
        VkDeviceQueueCreateInfo qInfo{};
        qInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        qInfo.queueFamilyIndex = idx;
        qInfo.queueCount = 1;
        qInfo.pQueuePriorities = &priority;

        devQInfo.emplace_back(qInfo);
    }

    // If you try to make an API call down the road which requires something be enabled,
    // you'll more than likely get a validation message telling you what to enable.
    // TODO: Come back and enable required features
    VkPhysicalDeviceFeatures deviceFeatures{};

    VkDeviceCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    info.queueCreateInfoCount = devQInfo.size();
    info.pQueueCreateInfos = devQInfo.data();
    info.pEnabledFeatures = &deviceFeatures;
    info.enabledExtensionCount = m_deviceExtensions.size();
    info.ppEnabledExtensionNames = m_deviceExtensions.data();

    if (m_enableValidation)
    {
        info.enabledLayerCount = m_validationLayers.size();
        info.ppEnabledLayerNames = m_validationLayers.data();
    }
    else
    {
        info.enabledLayerCount = 0;
    }

    if (vkCreateDevice(m_physicalDevice, &info, nullptr, &m_vkCtx.device) != VK_SUCCESS)
    {
        throw std::runtime_error("Failed to create the logical device!\n");
    }

    vkGetDeviceQueue(m_vkCtx.device, m_vkCtx.graphicsFamilyIdx, 0, &m_vkCtx.graphicsQueue);
    vkGetDeviceQueue(m_vkCtx.device, m_vkCtx.presentFamilyIdx, 0, &m_vkCtx.presentQueue);
}

void RenderBackend::CreateSemaphores()
{
    VkSemaphoreCreateInfo semaphoreCreateInfo{};
    semaphoreCreateInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

    // Synchronize access to rendering and presenting images (double buffered images)
    for (int i = 0; i < NUM_FRAME_DATA; i++)
    {
        vkCreateSemaphore(m_vkCtx.device, &semaphoreCreateInfo, nullptr, &m_acquireSemaphores[i]);
        vkCreateSemaphore(m_vkCtx.device, &semaphoreCreateInfo, nullptr, &m_renderCompleteSemaphores[i]);
    }
}

void RenderBackend::CreateQueryPool()
{
}

void RenderBackend::CreateCommandPool()
{
    // Because command buffers can be very flexible, we don't want to be
    // doing a lot of allocation while we're trying to render.
    // For this reason we create a pool to hold allocated command buffers.
    VkCommandPoolCreateInfo commandPoolCreateInfo{};
    commandPoolCreateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;

    // This allows the command buffer to be implicitly reset when vkBeginCommandBuffer is called.
    // You can also explicitly call vkResetCommandBuffer.
    commandPoolCreateInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;

    // We'll be building command buffers to send to the graphics queue
    commandPoolCreateInfo.queueFamilyIndex = m_vkCtx.graphicsFamilyIdx;

    if (vkCreateCommandPool(m_vkCtx.device, &commandPoolCreateInfo, nullptr, &m_commandPool) != VK_SUCCESS)
    {
        throw std::runtime_error("Failed to create the command pool!\n");
    }
}

void RenderBackend::CreateCommandBuffers()
{
    VkCommandBufferAllocateInfo commandBufferAllocateInfo{};
    commandBufferAllocateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    commandBufferAllocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    commandBufferAllocateInfo.commandPool = m_commandPool;
    commandBufferAllocateInfo.commandBufferCount = NUM_FRAME_DATA;

    // Allocating multiple command buffers at once
    vkAllocateCommandBuffers(m_vkCtx.device, &commandBufferAllocateInfo, m_commandBuffers.data());

    VkFenceCreateInfo fenceCreateInfo{};
    fenceCreateInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    // The first call to vkWaitForFences() returns immediately since the fence is already signaled
    fenceCreateInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

    // Create fences that we can use to wait for a given command buffer to be done on the GPU
    for (int i = 0; i < NUM_FRAME_DATA; i++)
    {
        vkCreateFence(m_vkCtx.device, &fenceCreateInfo, nullptr, &m_commandBufferFences[i]);
    }
}

void RenderBackend::CreateSwapChain()
{
    GPUInfo_t& gpu = m_vkCtx.gpu;

    VkSurfaceFormatKHR surfaceFormat = ChooseSurfaceFormat(gpu.surfaceFormats);
    const VkPresentModeKHR presentMode = ChoosePresentMode(gpu.presentModes);
    const VkExtent2D extent = ChooseSurfaceExtent(gpu.surfaceCaps);

    VkSwapchainCreateInfoKHR info{};
    info.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
    info.surface = m_surface;
    info.minImageCount = NUM_FRAME_DATA;
    info.imageFormat = surfaceFormat.format;
    info.imageColorSpace = surfaceFormat.colorSpace;
    info.imageExtent = extent;
    info.imageArrayLayers = 1;
    // VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT - This is a color image I'm rendering into
    // VK_IMAGE_USAGE_TRANSFER_SRC_BIT - I'll be copying this image somewhere (screenshot, postprocess)
    info.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;

    // If the graphics queue family and present family don't match
    // then we need to create the swapchain with different information.
    if (m_vkCtx.graphicsFamilyIdx != m_vkCtx.presentFamilyIdx)
    {
        const uint32_t indices[] = {
            static_cast<uint32_t>(m_vkCtx.graphicsFamilyIdx),
            static_cast<uint32_t>(m_vkCtx.presentFamilyIdx)
        };

        // There are only two sharing modes. This is the one to use if images are not exclusive to one queue.
        info.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
        info.queueFamilyIndexCount = std::size(indices);
        info.pQueueFamilyIndices = indices;
    }
    else
    {
        // If the indices are the same, then the queue can have exclusive access to the images.
        info.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    }

    // Leave image as is
    info.preTransform = VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR;
    info.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    info.presentMode = presentMode;

    // Is Vulkan allowed to discard operations outside of the renderable space?
    info.clipped = VK_TRUE;

    // Create swapchain
    vkCreateSwapchainKHR(m_vkCtx.device, &info, nullptr, &m_swapchain);

    // Save off swapchain details
    m_swapchainFormat = surfaceFormat.format;
    m_presentMode = presentMode;
    m_swapchainExtent = extent;

    // Retrieve the swapchain images from the device.
    // Note that VkImage is simply a handle like everything else.

    uint32_t numImages = 0;
    vkGetSwapchainImagesKHR(m_vkCtx.device, m_swapchain, &numImages, nullptr);
    if (numImages == 0)
    {
        throw std::runtime_error("vkGetSwapchainImagesKHR returned a zero image count.\n");
    }
    vkGetSwapchainImagesKHR(m_vkCtx.device, m_swapchain, &numImages, m_swapchainImages.data());
    if (numImages == 0)
    {
        throw std::runtime_error("vkGetSwapchainImagesKHR returned a zero image count.\n");
    }

    // Much like the logical device is an interface to the physical device,
    // image views are interfaces to actual images.  Think of it as this.
    // The image exists outside of you.  But the view is your personal view
    // ( how you perceive ) the image.
    for (uint32_t i = 0; i < NUM_FRAME_DATA; i++)
    {
        VkImageViewCreateInfo imageViewCreateInfo{};
        imageViewCreateInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        imageViewCreateInfo.image = m_swapchainImages[i];
        imageViewCreateInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
        imageViewCreateInfo.format = m_swapchainFormat;

        // We don't need to swizzle (swap around) any of the color channels
        imageViewCreateInfo.components.r = VK_COMPONENT_SWIZZLE_R;
        imageViewCreateInfo.components.g = VK_COMPONENT_SWIZZLE_G;
        imageViewCreateInfo.components.b = VK_COMPONENT_SWIZZLE_B;
        imageViewCreateInfo.components.a = VK_COMPONENT_SWIZZLE_A;

        // The subresourceRange field describes what the image's purpose is and which part of
        // the image should be accessed
        imageViewCreateInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        imageViewCreateInfo.subresourceRange.baseMipLevel = 0;
        imageViewCreateInfo.subresourceRange.levelCount = 1;
        imageViewCreateInfo.subresourceRange.baseArrayLayer = 0;
        imageViewCreateInfo.subresourceRange.layerCount = 1;
        imageViewCreateInfo.flags = 0;

        vkCreateImageView(m_vkCtx.device, &imageViewCreateInfo, nullptr, &m_swapchainViews[i]);
    }
}

void RenderBackend::CreateRenderTargets()
{
    // TODO: Had to skip this. Not sure how to implement this.
}

void RenderBackend::CreateRenderPass()
{
    VkAttachmentDescription colorAttachment{};
    colorAttachment.format = m_swapchainFormat;
    colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
    // Clear the framebuffer to black before drawing a new frame
    colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    // Rendered contents will be stored in memory and can be read later
    colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    colorAttachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

    VkAttachmentReference colorAttachmentRef{};
    // Index of attachment description array to refer
    colorAttachmentRef.attachment = 0;
    colorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    VkSubpassDescription subpass{};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = 1;
    // The index of the attachment in this array is directly referenced from the fragment
    // shader with the "layout(location = 0) out vec4 outColor" directive
    subpass.pColorAttachments = &colorAttachmentRef;

    VkSubpassDependency dependency{};
    dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
    dependency.dstSubpass = 0;
    dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dependency.srcAccessMask = 0;
    dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

    VkRenderPassCreateInfo renderPassInfo{};
    renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    renderPassInfo.attachmentCount = 1;
    renderPassInfo.pAttachments = &colorAttachment;
    renderPassInfo.subpassCount = 1;
    renderPassInfo.pSubpasses = &subpass;
    renderPassInfo.dependencyCount = 1;
    renderPassInfo.pDependencies = &dependency;

    vkCreateRenderPass(m_vkCtx.device, &renderPassInfo, nullptr, &m_vkCtx.renderPass);
}

void RenderBackend::CreatePipelineCache()
{
}

void RenderBackend::CreateGraphicsPipeline()
{
    // Vertex Input
    // TODO: This is hardcoded. Revisit this.
    VkPipelineVertexInputStateCreateInfo vertexInputState{};
    vertexInputState.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vertexInputState.vertexBindingDescriptionCount = 0;
    vertexInputState.pVertexBindingDescriptions = nullptr; // Optional
    vertexInputState.vertexAttributeDescriptionCount = 0;
    vertexInputState.pVertexAttributeDescriptions = nullptr; // Optional

    // Input Assembly
    VkPipelineInputAssemblyStateCreateInfo inputAssemblyState{};
    inputAssemblyState.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    inputAssemblyState.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    inputAssemblyState.primitiveRestartEnable = VK_FALSE;

    // Rasterization
    VkPipelineRasterizationStateCreateInfo rasterizationState{};
    rasterizationState.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rasterizationState.depthClampEnable = VK_FALSE;
    rasterizationState.rasterizerDiscardEnable = VK_FALSE;
    rasterizationState.polygonMode = VK_POLYGON_MODE_FILL;
    rasterizationState.lineWidth = 1.0f;
    rasterizationState.cullMode = VK_CULL_MODE_BACK_BIT;
    rasterizationState.frontFace = VK_FRONT_FACE_CLOCKWISE;
    rasterizationState.depthBiasEnable = VK_FALSE;

    //Multisampling
    VkPipelineMultisampleStateCreateInfo multisampleState{};
    multisampleState.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisampleState.sampleShadingEnable = VK_FALSE;
    multisampleState.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
    multisampleState.minSampleShading = 1.0f; // Optional

    // TODO: Depth / Stencil

    // Color Blend Attachment
    VkPipelineColorBlendAttachmentState colorBlendAttachment{};
    colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT |
        VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    colorBlendAttachment.blendEnable = VK_FALSE;

    // Color Blend
    VkPipelineColorBlendStateCreateInfo colorBlendState{};
    colorBlendState.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    colorBlendState.logicOpEnable = VK_FALSE;
    colorBlendState.logicOp = VK_LOGIC_OP_COPY;
    colorBlendState.attachmentCount = 1;
    colorBlendState.pAttachments = &colorBlendAttachment;

    // Read shader files and create shader module
    std::vector<char> vertShaderCode;
    ReadShaderFile(VERT_SHADER_PATH, vertShaderCode);
    VkShaderModule vertShaderModule = CreateShaderModule(vertShaderCode);
    std::vector<char> fragShaderCode;
    ReadShaderFile(FRAG_SHADER_PATH, fragShaderCode);
    VkShaderModule fragShaderModule = CreateShaderModule(fragShaderCode);

    std::vector<VkPipelineShaderStageCreateInfo> shaderStages;
    VkPipelineShaderStageCreateInfo stage{};
    stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stage.pName = "main";

    // Setup pipeline stage for vertex shader
    stage.stage = VK_SHADER_STAGE_VERTEX_BIT;
    stage.module = vertShaderModule;
    shaderStages.emplace_back(stage);

    // Setup pipeline stage for fragment shader
    stage.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    stage.module = fragShaderModule;
    shaderStages.emplace_back(stage);

    // A viewport is the region of the framebuffer that the output will be rendered to
    VkViewport viewport{};
    viewport.x = 0.0f;
    viewport.y = 0.0f;
    viewport.width = static_cast<float>(m_swapchainExtent.width);
    viewport.height = static_cast<float>(m_swapchainExtent.height);
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;

    VkRect2D scissor{};
    scissor.offset = {0, 0};
    scissor.extent = m_swapchainExtent;

    const std::vector<VkDynamicState> dynamicStates {
        VK_DYNAMIC_STATE_VIEWPORT,
        VK_DYNAMIC_STATE_SCISSOR
    };

    VkPipelineDynamicStateCreateInfo dynamicState{};
    dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dynamicState.dynamicStateCount = dynamicStates.size();
    dynamicState.pDynamicStates = dynamicStates.data();

    VkPipelineViewportStateCreateInfo viewportState{};
    viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewportState.viewportCount = 1;
    viewportState.scissorCount = 1;

    VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;

    vkCreatePipelineLayout(m_vkCtx.device, &pipelineLayoutInfo, nullptr, &m_pipelineLayout);

    VkGraphicsPipelineCreateInfo pipelineInfo{};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pipelineInfo.layout = m_pipelineLayout;
    pipelineInfo.renderPass = m_vkCtx.renderPass;
    pipelineInfo.subpass = 0;
    pipelineInfo.stageCount = shaderStages.size();
    pipelineInfo.pStages = shaderStages.data();
    pipelineInfo.pVertexInputState = &vertexInputState;
    pipelineInfo.pInputAssemblyState = &inputAssemblyState;
    pipelineInfo.pViewportState = &viewportState;
    pipelineInfo.pRasterizationState = &rasterizationState;
    pipelineInfo.pMultisampleState = &multisampleState;
    pipelineInfo.pDepthStencilState = nullptr; // Optional
    pipelineInfo.pColorBlendState = &colorBlendState;
    pipelineInfo.pDynamicState = &dynamicState;
    pipelineInfo.basePipelineHandle = VK_NULL_HANDLE; // Optional
    pipelineInfo.basePipelineIndex = -1; // Optional

    vkCreateGraphicsPipelines(m_vkCtx.device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &m_pipeline);

    vkDestroyShaderModule(m_vkCtx.device, vertShaderModule, nullptr);
    vkDestroyShaderModule(m_vkCtx.device, fragShaderModule, nullptr);
}

void RenderBackend::CreateFrameBuffers()
{
    for (size_t i = 0; i < m_swapchainImages.size(); i++) {
        const VkImageView attachments[] = {
            m_swapchainViews[i]
        };

        VkFramebufferCreateInfo framebufferInfo{};
        framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        framebufferInfo.renderPass = m_vkCtx.renderPass;
        framebufferInfo.attachmentCount = 1;
        framebufferInfo.pAttachments = attachments;
        framebufferInfo.width = m_swapchainExtent.width;
        framebufferInfo.height = m_swapchainExtent.height;
        framebufferInfo.layers = 1;

        vkCreateFramebuffer(m_vkCtx.device, &framebufferInfo, nullptr, &m_swapchainFramebuffers[i]);
    }
}

void RenderBackend::RecordCommandbuffer(const VkCommandBuffer& commandBuffer, const uint32_t imageIndex)
{
    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

    if (vkBeginCommandBuffer(commandBuffer, &beginInfo) != VK_SUCCESS)
    {
        throw std::runtime_error("Failed to begin recording command buffer!");
    }

    VkRenderPassBeginInfo renderPassBeginInfo{};
    renderPassBeginInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    renderPassBeginInfo.renderPass = m_vkCtx.renderPass;
    renderPassBeginInfo.framebuffer = m_swapchainFramebuffers[imageIndex];
    renderPassBeginInfo.renderArea.offset = {0, 0};
    renderPassBeginInfo.renderArea.extent = m_swapchainExtent;

    constexpr VkClearValue clearColor = {{{0.0f, 0.0f, 0.0f, 1.0f}}};
    renderPassBeginInfo.clearValueCount = 1;
    renderPassBeginInfo.pClearValues = &clearColor;

    vkCmdBeginRenderPass(commandBuffer, &renderPassBeginInfo, VK_SUBPASS_CONTENTS_INLINE);

    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipeline);

    VkViewport viewport{};
    viewport.x = 0.0f;
    viewport.y = 0.0f;
    viewport.width = static_cast<float>(m_swapchainExtent.width);
    viewport.height = static_cast<float>(m_swapchainExtent.height);
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;
    vkCmdSetViewport(commandBuffer, 0, 1, &viewport);

    VkRect2D scissor{};
    scissor.offset = {0, 0};
    scissor.extent = m_swapchainExtent;
    vkCmdSetScissor(commandBuffer, 0, 1, &scissor);

    vkCmdDraw(commandBuffer, 3, 1, 0, 0);

    vkCmdEndRenderPass(commandBuffer);

    if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS)
    {
        throw std::runtime_error("Failed to record command buffer!");
    }
}

void RenderBackend::DrawFrame()
{
    vkWaitForFences(m_vkCtx.device, 1, &m_commandBufferFences[m_currentFrame], VK_TRUE, UINT64_MAX);
    vkResetFences(m_vkCtx.device, 1, &m_commandBufferFences[m_currentFrame]);

    uint32_t imageIndex;
    vkAcquireNextImageKHR(
        m_vkCtx.device,m_swapchain, UINT64_MAX, m_acquireSemaphores[m_currentFrame], VK_NULL_HANDLE, &imageIndex);

    vkResetCommandBuffer(m_commandBuffers[m_currentFrame], 0);
    RecordCommandbuffer(m_commandBuffers[m_currentFrame], imageIndex);

    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;

    VkSemaphore waitSemaphores[] = {m_acquireSemaphores[m_currentFrame]};
    VkPipelineStageFlags waitStages[] = {VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};
    submitInfo.waitSemaphoreCount = 1;
    submitInfo.pWaitSemaphores = waitSemaphores;
    submitInfo.pWaitDstStageMask = waitStages;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &m_commandBuffers[m_currentFrame];

    VkSemaphore signalSemaphores[] = {m_renderCompleteSemaphores[m_currentFrame]};
    submitInfo.signalSemaphoreCount = 1;
    submitInfo.pSignalSemaphores = signalSemaphores;

    if (vkQueueSubmit(m_vkCtx.graphicsQueue, 1, &submitInfo, m_commandBufferFences[m_currentFrame]) != VK_SUCCESS)
    {
        throw std::runtime_error("Failed to submit draw command buffer!");
    }

    VkPresentInfoKHR presentInfo{};
    presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;

    presentInfo.waitSemaphoreCount = 1;
    presentInfo.pWaitSemaphores = signalSemaphores;

    VkSwapchainKHR swapChains[] = {m_swapchain};
    presentInfo.swapchainCount = std::size(swapChains);
    presentInfo.pSwapchains = swapChains;
    presentInfo.pImageIndices = &imageIndex;

    vkQueuePresentKHR(m_vkCtx.presentQueue, &presentInfo);

    m_currentFrame = ++m_currentFrame % NUM_FRAME_DATA;
}

VkExtent2D RenderBackend::ChooseSurfaceExtent(const VkSurfaceCapabilitiesKHR& caps)
{
    VkExtent2D extent;

    // The extent is typically the size of the window we created the surface from.
    // However if Vulkan returns -1 then simply substitute the window size.
    if ( caps.currentExtent.width == -1 )
    {
        int winWidth, winHeight;
        glfwGetWindowSize(m_window, &winWidth, &winHeight);
        extent.width = winWidth;
        extent.height = winHeight;
    }
    else
    {
        extent = caps.currentExtent;
    }

    return extent;
}

VkShaderModule RenderBackend::CreateShaderModule(const std::vector<char>& code)
{
    VkShaderModuleCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    createInfo.codeSize = code.size();
    createInfo.pCode = reinterpret_cast<const uint32_t*>(code.data());

    VkShaderModule shaderModule;
    vkCreateShaderModule(m_vkCtx.device, &createInfo, nullptr, &shaderModule);

    return shaderModule;
}
