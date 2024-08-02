#include "RenderBackend.h"

#include <iostream>
#include <cstring>
#include <unordered_set>

constexpr uint32_t SCR_WIDTH = 1280;
constexpr uint32_t SCR_HEIGHT = 720;

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
    // Favor looking for mailbox mode
    constexpr VkPresentModeKHR desiredMode = VK_PRESENT_MODE_MAILBOX_KHR;

    for (const VkPresentModeKHR& mode : modes)
    {
        if (mode == desiredMode)
        {
            return desiredMode;
        }
    }

    // If we couldn't find mailbox, then default to FIFO which is always available
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

RenderBackend::RenderBackend() : m_window(nullptr),
                                 m_instance(),
                                 m_surface(),
                                 m_presentMode(),
                                 m_physicalDevice(),
                                 m_enableValidation(true),
                                 m_acquireSemaphores(NUM_FRAME_DATA),
                                 m_renderCompleteSemaphores(NUM_FRAME_DATA),
                                 m_commandPool(),
                                 m_commandBuffers(NUM_FRAME_DATA),
                                 m_commandBufferFences(NUM_FRAME_DATA),
                                 m_swapchain(),
                                 m_swapchainFormat(),
                                 m_swapchainExtent(),
                                 m_swapchainImages(NUM_FRAME_DATA),
                                 m_swapchainViews(NUM_FRAME_DATA)
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

    // Create Frame Buffers
    CreateFrameBuffers();

    // Init RenderProg Manager
    // renderProgManager.Init();

    // Init Vertex Cache
    // vertexCache.Init( vkcontext.gpu.props.limits.minUniformBufferOffsetAlignment );
}

void RenderBackend::Shutdown()
{
    for (uint32_t i = 0; i < NUM_FRAME_DATA; i++)
    {
        vkDestroyImageView(m_vkContext.device, m_swapchainViews[i], nullptr);
    }
    vkDestroySwapchainKHR(m_vkContext.device, m_swapchain, nullptr);

    // Destroy fences
    for (const VkFence& fence : m_commandBufferFences)
    {
        vkDestroyFence(m_vkContext.device, fence, nullptr);
    }

    // TODO: Is it necessary to release command buffers? Validation layer doesn't complain

    vkDestroyCommandPool(m_vkContext.device, m_commandPool, nullptr);

    // Destroy semaphores
    for (const VkSemaphore& semaphore : m_acquireSemaphores)
    {
        vkDestroySemaphore(m_vkContext.device, semaphore, nullptr);
    }
    for (const VkSemaphore& semaphore : m_renderCompleteSemaphores)
    {
        vkDestroySemaphore(m_vkContext.device, semaphore, nullptr);
    }

    // Destroy logical device
    vkDestroyDevice(m_vkContext.device, nullptr);
    // Destroy window surface
    vkDestroySurfaceKHR(m_instance, m_surface, nullptr);
    vkDestroyInstance(m_instance, nullptr);

    glfwDestroyWindow(m_window);
    glfwTerminate();
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
            m_vkContext.graphicsFamilyIdx = graphicsIdx;
            m_vkContext.presentFamilyIdx = presentIdx;
            m_physicalDevice = gpu.device;
            m_vkContext.gpu = gpu;
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
    uniqueIdx.insert(m_vkContext.graphicsFamilyIdx);
    uniqueIdx.insert(m_vkContext.presentFamilyIdx);

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

    if (vkCreateDevice(m_physicalDevice, &info, nullptr, &m_vkContext.device) != VK_SUCCESS)
    {
        throw std::runtime_error("Failed to create the logical device!\n");
    }

    vkGetDeviceQueue(m_vkContext.device, m_vkContext.graphicsFamilyIdx, 0, &m_vkContext.graphicsQueue);
    vkGetDeviceQueue(m_vkContext.device, m_vkContext.presentFamilyIdx, 0, &m_vkContext.presentQueue);
}

void RenderBackend::CreateSemaphores()
{
    VkSemaphoreCreateInfo semaphoreCreateInfo{};
    semaphoreCreateInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

    // Synchronize access to rendering and presenting images (double buffered images)
    for (int i = 0; i < NUM_FRAME_DATA; i++)
    {
        vkCreateSemaphore(m_vkContext.device, &semaphoreCreateInfo, nullptr, &m_acquireSemaphores[i]);
        vkCreateSemaphore(m_vkContext.device, &semaphoreCreateInfo, nullptr, &m_renderCompleteSemaphores[i]);
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
    commandPoolCreateInfo.queueFamilyIndex = m_vkContext.graphicsFamilyIdx;

    if (vkCreateCommandPool(m_vkContext.device, &commandPoolCreateInfo, nullptr, &m_commandPool) != VK_SUCCESS)
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
    vkAllocateCommandBuffers(m_vkContext.device, &commandBufferAllocateInfo, m_commandBuffers.data());

    VkFenceCreateInfo fenceCreateInfo{};
    fenceCreateInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;

    // Create fences that we can use to wait for a given command buffer to be done on the GPU
    for (int i = 0; i < NUM_FRAME_DATA; i++)
    {
        vkCreateFence(m_vkContext.device, &fenceCreateInfo, nullptr, &m_commandBufferFences[i]);
    }
}

void RenderBackend::CreateSwapChain()
{
    GPUInfo_t& gpu = m_vkContext.gpu;

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
    if (m_vkContext.graphicsFamilyIdx != m_vkContext.presentFamilyIdx)
    {
        const uint32_t indices[] = {
            static_cast<uint32_t>(m_vkContext.graphicsFamilyIdx),
            static_cast<uint32_t>(m_vkContext.presentFamilyIdx)
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
    vkCreateSwapchainKHR(m_vkContext.device, &info, nullptr, &m_swapchain);

    // Save off swapchain details
    m_swapchainFormat = surfaceFormat.format;
    m_presentMode = presentMode;
    m_swapchainExtent = extent;

    // Retrieve the swapchain images from the device.
    // Note that VkImage is simply a handle like everything else.

    uint32_t numImages = 0;
    vkGetSwapchainImagesKHR(m_vkContext.device, m_swapchain, &numImages, nullptr);
    if (numImages == 0)
    {
        throw std::runtime_error("vkGetSwapchainImagesKHR returned a zero image count.\n");
    }
    vkGetSwapchainImagesKHR(m_vkContext.device, m_swapchain, &numImages, m_swapchainImages.data());
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

        vkCreateImageView(m_vkContext.device, &imageViewCreateInfo, nullptr, &m_swapchainViews[i]);
    }
}

void RenderBackend::CreateRenderTargets()
{
    // TODO: Had to skip this. Not sure how to implement this.
}

void RenderBackend::CreateRenderPass()
{
}

void RenderBackend::CreatePipelineCache()
{
}

void RenderBackend::CreateGraphicsPipeline()
{

}

void RenderBackend::CreateFrameBuffers()
{
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
