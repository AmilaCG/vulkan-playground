#include "RenderBackend.h"

#include <iostream>
#include <fstream>
#include <cstring>
#include <unordered_set>
#include <chrono>
#include <glm/gtc/matrix_transform.hpp>
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
#define TINYOBJLOADER_IMPLEMENTATION
#include <tiny_obj_loader.h>

namespace
{
constexpr uint32_t SCR_WIDTH = 1280;
constexpr uint32_t SCR_HEIGHT = 720;

const std::string VERT_SHADER_PATH = "vert.spv";
const std::string FRAG_SHADER_PATH = "frag.spv";

const char* TEXTURE_IMG_PATH = "Assets/Textures/texture.jpg";

const char* MODEL_PATH = "Assets/Models/viking_room.obj";
const char* MODEL_TEX_PATH = "Assets/Textures/viking_room.png";

const char* g_debugInstanceExtensions[] = {
    VK_EXT_DEBUG_REPORT_EXTENSION_NAME
};

const char* g_deviceExtensions[] = {
    VK_KHR_SWAPCHAIN_EXTENSION_NAME
};

const char* g_validationLayers[] = {
    "VK_LAYER_KHRONOS_validation", "VK_LAYER_LUNARG_monitor"
};

VulkanContext_t g_vkCtx{};

void ValidateValidationLayers()
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

bool CheckPhysicalDeviceExtensionSupport(GPUInfo_t& gpu, std::vector<const char*>& requiredExt)
{
    const size_t required = requiredExt.size();
    size_t available = 0;

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

VkSurfaceFormatKHR ChooseSurfaceFormat(std::vector<VkSurfaceFormatKHR>& formats)
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

VkPresentModeKHR ChoosePresentMode(std::vector<VkPresentModeKHR>& modes)
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

void ReadShaderFile(const std::string& filename, std::vector<char>& buffer)
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

VkShaderModule CreateShaderModule(const std::vector<char>& code)
{
    VkShaderModuleCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    createInfo.codeSize = code.size();
    createInfo.pCode = reinterpret_cast<const uint32_t*>(code.data());

    VkShaderModule shaderModule;
    vkCreateShaderModule(g_vkCtx.device, &createInfo, nullptr, &shaderModule);

    return shaderModule;
}

void OnFramebufferResize(GLFWwindow* window, int width, int height)
{
    const auto renderer = static_cast<RenderBackend*>(glfwGetWindowUserPointer(window));
    renderer->SetFramebufferResizeFlag(true);
}

uint32_t FindMemoryType(const uint32_t memTypeBitsRequirement, const VkMemoryPropertyFlags requiredProperties)
{
    VkPhysicalDeviceMemoryProperties memoryProperties{};
    vkGetPhysicalDeviceMemoryProperties(g_vkCtx.gpu.device, &memoryProperties);

    for (uint32_t memIndex = 0; memIndex < memoryProperties.memoryTypeCount; memIndex++)
    {
        // Keep shifting 1 to left and compare it with the corresponding bit index (memIndex) of
        // memoryTypeBitsRequirement value. isRequiredMemoryType = true when both are set.
        // For example, when memTypeBitsRequirement = 1921 (decimal) / 0000 0111 1000 0001 (binary),
        // isRequiredMemoryType = true when memIndex is 1, 7, 8, 9 and 10.
        const bool isRequiredMemoryType = memTypeBitsRequirement & (1 << memIndex);

        const bool hasRequiredProperties =
            (memoryProperties.memoryTypes[memIndex].propertyFlags & requiredProperties) == requiredProperties;

        if (isRequiredMemoryType && hasRequiredProperties)
        {
            return memIndex;
        }
    }

    throw std::runtime_error("Failed to find suitable memory type!");
}

void CreateBuffer(
    VkDeviceSize size,
    VkBufferUsageFlags usage,
    VkMemoryPropertyFlags properties,
    VkBuffer& buffer,
    VkDeviceMemory& bufferMemory)
{
    VkBufferCreateInfo bufferInfo{};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = size;
    bufferInfo.usage = usage;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    if (vkCreateBuffer(g_vkCtx.device, &bufferInfo, nullptr, &buffer) != VK_SUCCESS)
    {
        throw std::runtime_error("Failed to create vertex buffer!");
    }

    VkMemoryRequirements memoryRequirements;
    vkGetBufferMemoryRequirements(g_vkCtx.device, buffer, &memoryRequirements);

    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memoryRequirements.size;
    allocInfo.memoryTypeIndex = FindMemoryType(memoryRequirements.memoryTypeBits, properties);

    if (vkAllocateMemory(g_vkCtx.device, &allocInfo, nullptr, &bufferMemory) != VK_SUCCESS)
    {
        throw std::runtime_error("Failed to allocate buffer memory!");
    }

    if (vkBindBufferMemory(g_vkCtx.device, buffer, bufferMemory, 0) != VK_SUCCESS)
    {
        throw std::runtime_error("Failed binding to buffer memory!");
    }
}

void CreateImage(
    uint32_t width,
    uint32_t height,
    VkFormat format,
    VkImageTiling tiling,
    VkImageUsageFlags usage,
    VkMemoryPropertyFlags properties,
    VkImage& image,
    VkDeviceMemory& imageMemory)
{
    VkImageCreateInfo imageInfo{};
    imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageInfo.imageType = VK_IMAGE_TYPE_2D;
    imageInfo.extent.width = width;
    imageInfo.extent.height = height;
    imageInfo.extent.depth = 1;
    imageInfo.mipLevels = 1;
    imageInfo.arrayLayers = 1;
    imageInfo.format = format;
    imageInfo.tiling = tiling;
    imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    imageInfo.usage = usage;
    imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    if (vkCreateImage(g_vkCtx.device, &imageInfo, nullptr, &image) != VK_SUCCESS)
    {
        throw std::runtime_error("Failed to create image!");
    }

    VkMemoryRequirements memRequirements;
    vkGetImageMemoryRequirements(g_vkCtx.device, image, &memRequirements);

    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memRequirements.size;
    allocInfo.memoryTypeIndex = FindMemoryType(memRequirements.memoryTypeBits, properties);

    if (vkAllocateMemory(g_vkCtx.device, &allocInfo, nullptr, &imageMemory) != VK_SUCCESS)
    {
        throw std::runtime_error("Failed to allocate image memory!");
    }

    vkBindImageMemory(g_vkCtx.device, image, imageMemory, 0);
}

VkFormat FindSupportedFormat(
    const std::vector<VkFormat>& candidates,
    VkImageTiling tiling,
    VkFormatFeatureFlags features)
{
    for (VkFormat format : candidates)
    {
        VkFormatProperties props;
        vkGetPhysicalDeviceFormatProperties(g_vkCtx.gpu.device, format, &props);

        if (tiling == VK_IMAGE_TILING_LINEAR && (props.linearTilingFeatures & features) == features)
        {
            return format;
        }
        if (tiling == VK_IMAGE_TILING_OPTIMAL && (props.optimalTilingFeatures & features) == features)
        {
            return format;
        }
    }

    throw std::runtime_error("Failed to find supported format!");
}

VkFormat FindDepthFormat()
{
    return FindSupportedFormat(
        {VK_FORMAT_D32_SFLOAT, VK_FORMAT_D32_SFLOAT_S8_UINT, VK_FORMAT_D24_UNORM_S8_UINT},
        VK_IMAGE_TILING_OPTIMAL,
        VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT
    );
}

bool HasStencilComponent(VkFormat format)
{
    return format == VK_FORMAT_D32_SFLOAT_S8_UINT || format == VK_FORMAT_D24_UNORM_S8_UINT;
}
} // namespace

RenderBackend::RenderBackend()
{
#ifdef VALIDATION_OFF
    g_vkCtx.enableValidation = false;
#else
    g_vkCtx.enableValidation = true;
#endif
}

RenderBackend::~RenderBackend() = default;

void RenderBackend::Init()
{
    if (!WindowInit())
    {
        throw std::runtime_error("Unable to initialize the GLFW window");
    }

    CreateInstance();
    CreateSurface();
    SelectSuitablePhysicalDevice();
    CreateLogicalDeviceAndQueues();
    CreateSemaphores();
    CreateCommandPool();
    CreateSwapChain();
    CreateDepthResources();
    CreateTextureImage();
    CreateTextureImageView();
    CreateTextureSampler();
    LoadModel();
    CreateVertexBuffer();
    CreateIndexBuffer();
    CreateUniformBuffers();
    CreateDescriptorSetLayout();
    CreateDescriptorPool();
    CreateDescriptorSets();
    CreateCommandBuffers();
    CreateRenderPass();
    CreateGraphicsPipeline();
    CreateFrameBuffers();
}

void RenderBackend::Shutdown()
{
    CleanupSwapchain();

    vkDestroySampler(g_vkCtx.device, m_textureSampler, nullptr);
    vkDestroyImageView(g_vkCtx.device, m_textureImageView, nullptr);

    vkDestroyImage(g_vkCtx.device, m_textureImage, nullptr);
    vkFreeMemory(g_vkCtx.device, m_textureImageMemory, nullptr);

    for (uint32_t i = 0; i < FRAMES_IN_FLIGHT; i++)
    {
        vkDestroyBuffer(g_vkCtx.device, m_uniformBuffers[i], nullptr);
        vkFreeMemory(g_vkCtx.device, m_uniformBufferMemories[i], nullptr);
    }

    vkDestroyDescriptorPool(g_vkCtx.device, m_descriptorPool, nullptr);

    vkDestroyDescriptorSetLayout(g_vkCtx.device, m_descriptorSetLayout, nullptr);

    vkDestroyBuffer(g_vkCtx.device, m_indexBuffer, nullptr);
    vkFreeMemory(g_vkCtx.device, m_indexBufferMemory, nullptr);

    vkDestroyBuffer(g_vkCtx.device, m_vertexBuffer, nullptr);
    vkFreeMemory(g_vkCtx.device, m_vertexBufferMemory, nullptr);

    vkDestroyPipeline(g_vkCtx.device, m_pipeline, nullptr);
    vkDestroyPipelineLayout(g_vkCtx.device, m_pipelineLayout, nullptr);
    vkDestroyRenderPass(g_vkCtx.device, g_vkCtx.renderPass, nullptr);

    // Destroy fences
    for (const VkFence& fence : m_commandBufferFences)
    {
        vkDestroyFence(g_vkCtx.device, fence, nullptr);
    }

    // TODO: Is it necessary to release command buffers? Validation layer doesn't complain
    vkDestroyCommandPool(g_vkCtx.device, m_commandPool, nullptr);

    // Destroy semaphores
    for (const VkSemaphore& semaphore : m_acquireSemaphores)
    {
        vkDestroySemaphore(g_vkCtx.device, semaphore, nullptr);
    }
    for (const VkSemaphore& semaphore : m_renderCompleteSemaphores)
    {
        vkDestroySemaphore(g_vkCtx.device, semaphore, nullptr);
    }

    // Destroy logical device
    vkDestroyDevice(g_vkCtx.device, nullptr);
    // Destroy window surface
    vkDestroySurfaceKHR(g_vkCtx.instance, g_vkCtx.surface, nullptr);
    vkDestroyInstance(g_vkCtx.instance, nullptr);

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
    vkDeviceWaitIdle(g_vkCtx.device);
}

// Although many drivers and platforms trigger VK_ERROR_OUT_OF_DATE_KHR automatically after a window resize,
// it is not guaranteed to happen. Therefore we use this flag as a backup.
void RenderBackend::SetFramebufferResizeFlag(const bool resized)
{
    m_frameBufferResized = resized;
}

bool RenderBackend::WindowInit()
{
    glfwInit();

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    m_window = glfwCreateWindow(SCR_WIDTH, SCR_HEIGHT, "Vulkan Playground", nullptr, nullptr);
    glfwSetWindowUserPointer(m_window, this);
    glfwSetFramebufferSizeCallback(m_window, OnFramebufferResize);

    return m_window != nullptr;
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

    if (g_vkCtx.enableValidation)
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

    if (vkCreateInstance(&createInfo, nullptr, &g_vkCtx.instance) != VK_SUCCESS)
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

    if (vkCreateWin32SurfaceKHR(g_vkCtx.instance, &createInfo, nullptr, &g_vkCtx.surface) != VK_SUCCESS)
    {
        throw std::runtime_error("Failed to create window surface!\n");
    }
}

void RenderBackend::SelectSuitablePhysicalDevice()
{
    uint32_t numDevices = 0;
    vkEnumeratePhysicalDevices(g_vkCtx.instance, &numDevices, nullptr);
    if (numDevices == 0)
    {
        throw std::runtime_error("vkEnumeratePhysicalDevices returned zero devices.\n");
    }

    std::vector<VkPhysicalDevice> devices(numDevices);
    if (vkEnumeratePhysicalDevices(g_vkCtx.instance, &numDevices, devices.data()) != VK_SUCCESS)
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
        vkGetPhysicalDeviceSurfaceCapabilitiesKHR(gpu.device, g_vkCtx.surface, &gpu.surfaceCaps);

        {
            // Get the supported surface formats. This includes image format and color space.
            uint32_t numFormats;
            vkGetPhysicalDeviceSurfaceFormatsKHR(gpu.device, g_vkCtx.surface, &numFormats, nullptr);
            if (numFormats == 0)
            {
                throw std::runtime_error("vkGetPhysicalDeviceSurfaceFormatsKHR returned zero formats.\n");
            }

            gpu.surfaceFormats.resize(numFormats);
            vkGetPhysicalDeviceSurfaceFormatsKHR(gpu.device, g_vkCtx.surface, &numFormats, gpu.surfaceFormats.data());
            if (numFormats == 0)
            {
                throw std::runtime_error("vkGetPhysicalDeviceSurfaceFormatsKHR returned zero formats.\n");
            }
        }

        {
            // Get supported presentation modes
            uint32_t numPresentModes;
            vkGetPhysicalDeviceSurfacePresentModesKHR(gpu.device, g_vkCtx.surface, &numPresentModes, nullptr);
            if (numPresentModes == 0)
            {
                throw std::runtime_error("vkGetPhysicalDeviceSurfacePresentModesKHR returned zero present modes.\n");
            }

            gpu.presentModes.resize(numPresentModes);
            vkGetPhysicalDeviceSurfacePresentModesKHR(gpu.device, g_vkCtx.surface, &numPresentModes, gpu.presentModes.data());
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
            vkGetPhysicalDeviceSurfaceSupportKHR(gpu.device, i, g_vkCtx.surface, &supportPresentation);
            if (supportPresentation)
            {
                presentIdx = i;
                break;
            }
        }

        // Did we find a device supporting both graphics and present
        if (graphicsIdx >= 0 && presentIdx >= 0)
        {
            g_vkCtx.graphicsFamilyIdx = graphicsIdx;
            g_vkCtx.presentFamilyIdx = presentIdx;
            g_vkCtx.gpu.device = gpu.device;
            g_vkCtx.gpu = gpu;
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
    uniqueIdx.insert(g_vkCtx.graphicsFamilyIdx);
    uniqueIdx.insert(g_vkCtx.presentFamilyIdx);

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
    deviceFeatures.samplerAnisotropy = VK_TRUE;

    VkDeviceCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    info.queueCreateInfoCount = devQInfo.size();
    info.pQueueCreateInfos = devQInfo.data();
    info.pEnabledFeatures = &deviceFeatures;
    info.enabledExtensionCount = m_deviceExtensions.size();
    info.ppEnabledExtensionNames = m_deviceExtensions.data();

    if (g_vkCtx.enableValidation)
    {
        info.enabledLayerCount = m_validationLayers.size();
        info.ppEnabledLayerNames = m_validationLayers.data();
    }
    else
    {
        info.enabledLayerCount = 0;
    }

    if (vkCreateDevice(g_vkCtx.gpu.device, &info, nullptr, &g_vkCtx.device) != VK_SUCCESS)
    {
        throw std::runtime_error("Failed to create the logical device!\n");
    }

    vkGetDeviceQueue(g_vkCtx.device, g_vkCtx.graphicsFamilyIdx, 0, &g_vkCtx.graphicsQueue);
    vkGetDeviceQueue(g_vkCtx.device, g_vkCtx.presentFamilyIdx, 0, &g_vkCtx.presentQueue);
}

void RenderBackend::CreateSemaphores()
{
    VkSemaphoreCreateInfo semaphoreCreateInfo{};
    semaphoreCreateInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

    // Synchronize access to rendering and presenting images (double buffered images)
    for (int i = 0; i < FRAMES_IN_FLIGHT; i++)
    {
        vkCreateSemaphore(g_vkCtx.device, &semaphoreCreateInfo, nullptr, &m_acquireSemaphores[i]);
        vkCreateSemaphore(g_vkCtx.device, &semaphoreCreateInfo, nullptr, &m_renderCompleteSemaphores[i]);
    }
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
    commandPoolCreateInfo.queueFamilyIndex = g_vkCtx.graphicsFamilyIdx;

    if (vkCreateCommandPool(g_vkCtx.device, &commandPoolCreateInfo, nullptr, &m_commandPool) != VK_SUCCESS)
    {
        throw std::runtime_error("Failed to create the command pool!\n");
    }
}

void RenderBackend::CreateDepthResources()
{
    const VkFormat depthFormat = FindDepthFormat();
    CreateImage(m_swapchainExtent.width, m_swapchainExtent.height,
        depthFormat,
        VK_IMAGE_TILING_OPTIMAL,
        VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        m_depthImage,
        m_depthImageMemory);

    m_depthImageView = CreateImageView(m_depthImage, depthFormat, VK_IMAGE_ASPECT_DEPTH_BIT);
}

void RenderBackend::CreateTextureImage()
{
    int texWidth, texHeight, texChannels;
    stbi_uc* pixels = stbi_load(MODEL_TEX_PATH, &texWidth, &texHeight, &texChannels, STBI_rgb_alpha);
    const VkDeviceSize imageSize = texWidth * texHeight * 4;

    if (!pixels)
    {
        throw std::runtime_error("Failed to load texture image!");
    }

    VkBuffer stagingBuffer;
    VkDeviceMemory stagingBufferMemory;

    CreateBuffer(imageSize,
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        stagingBuffer,
        stagingBufferMemory);

    void* data;
    vkMapMemory(g_vkCtx.device, stagingBufferMemory, 0, imageSize, 0, &data);
    memcpy(data, pixels, imageSize);
    vkUnmapMemory(g_vkCtx.device, stagingBufferMemory);

    stbi_image_free(pixels);

    CreateImage(texWidth,
        texHeight,
        VK_FORMAT_R8G8B8A8_SRGB,
        VK_IMAGE_TILING_OPTIMAL,
        VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        m_textureImage,
        m_textureImageMemory);

    TransitionImageLayout(m_textureImage,
        VK_FORMAT_R8G8B8A8_SRGB,
        VK_IMAGE_LAYOUT_UNDEFINED,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);

    CopyBufferToImage(stagingBuffer,
        m_textureImage,
        static_cast<uint32_t>(texWidth),
        static_cast<uint32_t>(texHeight));

    TransitionImageLayout(m_textureImage,
        VK_FORMAT_R8G8B8A8_SRGB,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

    vkDestroyBuffer(g_vkCtx.device, stagingBuffer, nullptr);
    vkFreeMemory(g_vkCtx.device, stagingBufferMemory, nullptr);
}

void RenderBackend::CreateTextureImageView()
{
    m_textureImageView = CreateImageView(m_textureImage, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_ASPECT_COLOR_BIT);
}

void RenderBackend::LoadModel()
{
    tinyobj::attrib_t attrib;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> materials;
    std::string warn;
    std::string err;

    if (!tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, MODEL_PATH))
    {
        throw std::runtime_error(warn + err);
    }

    for (const auto& shape : shapes)
    {
        for (const auto& index : shape.mesh.indices)
        {
            Vertex_t vertex{};

            vertex.pos = {
                attrib.vertices[3 * index.vertex_index + 0],
                attrib.vertices[3 * index.vertex_index + 1],
                attrib.vertices[3 * index.vertex_index + 2]
            };

            vertex.texCoord = {
                attrib.texcoords[2 * index.texcoord_index + 0],
                1.0f - attrib.texcoords[2 * index.texcoord_index + 1]
            };

            vertex.color = {1.0f, 1.0f, 1.0f};

            m_vertices.push_back(vertex);
            m_indices.push_back(m_indices.size());
        }
    }
}

void RenderBackend::CreateVertexBuffer()
{
    const VkDeviceSize bufferSize = sizeof(m_vertices[0]) * m_vertices.size();

    VkBuffer stagingBuffer;
    VkDeviceMemory stagingBufferMemory;
    // Staging buffer
    CreateBuffer(bufferSize,
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        stagingBuffer,
        stagingBufferMemory);

    void* data;
    vkMapMemory(g_vkCtx.device, stagingBufferMemory, 0, bufferSize, 0, &data);
    memcpy(data, m_vertices.data(), bufferSize);
    vkUnmapMemory(g_vkCtx.device, stagingBufferMemory);

    CreateBuffer(bufferSize,
        VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        m_vertexBuffer,
        m_vertexBufferMemory);

    CopyBuffer(stagingBuffer, m_vertexBuffer, bufferSize, m_commandPool);

    vkDestroyBuffer(g_vkCtx.device, stagingBuffer, nullptr);
    vkFreeMemory(g_vkCtx.device, stagingBufferMemory, nullptr);
}

void RenderBackend::CreateIndexBuffer()
{
    const VkDeviceSize bufferSize = sizeof(m_indices[0]) * m_indices.size();

    VkBuffer stagingBuffer;
    VkDeviceMemory stagingBufferMemory;
    CreateBuffer(bufferSize,
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        stagingBuffer,
        stagingBufferMemory);

    void* data;
    vkMapMemory(g_vkCtx.device, stagingBufferMemory, 0, bufferSize, 0, &data);
    memcpy(data, m_indices.data(), bufferSize);
    vkUnmapMemory(g_vkCtx.device, stagingBufferMemory);

    CreateBuffer(bufferSize,
        VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        m_indexBuffer,
        m_indexBufferMemory);

    CopyBuffer(stagingBuffer, m_indexBuffer, bufferSize, m_commandPool);

    vkDestroyBuffer(g_vkCtx.device, stagingBuffer, nullptr);
    vkFreeMemory(g_vkCtx.device, stagingBufferMemory, nullptr);
}

void RenderBackend::CreateUniformBuffers()
{
    for (uint32_t i = 0; i < FRAMES_IN_FLIGHT; i++)
    {
        constexpr VkDeviceSize bufferSize = sizeof(UniformBufferObject_t);
        CreateBuffer(bufferSize,
            VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            m_uniformBuffers[i],
            m_uniformBufferMemories[i]);

        vkMapMemory(g_vkCtx.device, m_uniformBufferMemories[i], 0, bufferSize, 0, &m_uniformBufferMappings[i]);
    }
}

void RenderBackend::CreateDescriptorPool()
{
    VkDescriptorPoolSize poolSizeUniform{};
    poolSizeUniform.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    poolSizeUniform.descriptorCount = FRAMES_IN_FLIGHT;

    VkDescriptorPoolSize poolSizeSampler{};
    poolSizeSampler.type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    poolSizeSampler.descriptorCount = FRAMES_IN_FLIGHT;

    std::array poolSizes = {poolSizeUniform, poolSizeSampler};

    VkDescriptorPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.poolSizeCount = poolSizes.size();
    poolInfo.pPoolSizes = poolSizes.data();
    poolInfo.maxSets = FRAMES_IN_FLIGHT;

    if (vkCreateDescriptorPool(g_vkCtx.device, &poolInfo, nullptr, &m_descriptorPool) != VK_SUCCESS)
    {
        throw std::runtime_error("Failed to create descriptor pool!");
    }
}

void RenderBackend::CreateDescriptorSets()
{
    const std::vector<VkDescriptorSetLayout> layouts(FRAMES_IN_FLIGHT, m_descriptorSetLayout);
    VkDescriptorSetAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = m_descriptorPool;
    allocInfo.descriptorSetCount = static_cast<uint32_t>(FRAMES_IN_FLIGHT);
    allocInfo.pSetLayouts = layouts.data();

    if (vkAllocateDescriptorSets(g_vkCtx.device, &allocInfo, m_descriptorSets.data()) != VK_SUCCESS)
    {
        throw std::runtime_error("Failed to allocate descriptor sets!");
    }

    for (uint32_t i = 0; i < FRAMES_IN_FLIGHT; i++)
    {
        VkDescriptorBufferInfo bufferInfo{};
        bufferInfo.buffer = m_uniformBuffers[i];
        bufferInfo.offset = 0;
        bufferInfo.range = sizeof(UniformBufferObject_t);

        VkDescriptorImageInfo imageInfo{};
        imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        imageInfo.imageView = m_textureImageView;
        imageInfo.sampler = m_textureSampler;

        VkWriteDescriptorSet descriptorWriteUniform{};
        descriptorWriteUniform.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptorWriteUniform.dstSet = m_descriptorSets[i];
        descriptorWriteUniform.dstBinding = 0;
        descriptorWriteUniform.dstArrayElement = 0;
        descriptorWriteUniform.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        descriptorWriteUniform.descriptorCount = 1;
        descriptorWriteUniform.pBufferInfo = &bufferInfo;

        VkWriteDescriptorSet descriptorWriteSampler{};
        descriptorWriteSampler.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptorWriteSampler.dstSet = m_descriptorSets[i];
        descriptorWriteSampler.dstBinding = 1;
        descriptorWriteSampler.dstArrayElement = 0;
        descriptorWriteSampler.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        descriptorWriteSampler.descriptorCount = 1;
        descriptorWriteSampler.pImageInfo = &imageInfo;

        std::array descriptorWrites = {descriptorWriteUniform, descriptorWriteSampler};

        vkUpdateDescriptorSets(g_vkCtx.device, descriptorWrites.size(), descriptorWrites.data(), 0, nullptr);
    }
}

void RenderBackend::CreateCommandBuffers()
{
    VkCommandBufferAllocateInfo commandBufferAllocateInfo{};
    commandBufferAllocateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    commandBufferAllocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    commandBufferAllocateInfo.commandPool = m_commandPool;
    commandBufferAllocateInfo.commandBufferCount = FRAMES_IN_FLIGHT;

    // Allocating multiple command buffers at once
    vkAllocateCommandBuffers(g_vkCtx.device, &commandBufferAllocateInfo, m_commandBuffers.data());

    VkFenceCreateInfo fenceCreateInfo{};
    fenceCreateInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    // The first call to vkWaitForFences() returns immediately since the fence is already signaled
    fenceCreateInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

    // Create fences that we can use to wait for a given command buffer to be done on the GPU
    for (int i = 0; i < FRAMES_IN_FLIGHT; i++)
    {
        vkCreateFence(g_vkCtx.device, &fenceCreateInfo, nullptr, &m_commandBufferFences[i]);
    }
}

void RenderBackend::CreateSwapChain()
{
    GPUInfo_t& gpu = g_vkCtx.gpu;

    VkSurfaceFormatKHR surfaceFormat = ChooseSurfaceFormat(gpu.surfaceFormats);
    const VkPresentModeKHR presentMode = ChoosePresentMode(gpu.presentModes);
    const VkExtent2D extent = ChooseSurfaceExtent(gpu.surfaceCaps);

    VkSwapchainCreateInfoKHR info{};
    info.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
    info.surface = g_vkCtx.surface;
    info.minImageCount = FRAMES_IN_FLIGHT;
    info.imageFormat = surfaceFormat.format;
    info.imageColorSpace = surfaceFormat.colorSpace;
    info.imageExtent = extent;
    info.imageArrayLayers = 1;
    // VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT - This is a color image I'm rendering into
    // VK_IMAGE_USAGE_TRANSFER_SRC_BIT - I'll be copying this image somewhere (screenshot, postprocess)
    info.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;

    // If the graphics queue family and present family don't match
    // then we need to create the swapchain with different information.
    if (g_vkCtx.graphicsFamilyIdx != g_vkCtx.presentFamilyIdx)
    {
        const uint32_t indices[] = {
            static_cast<uint32_t>(g_vkCtx.graphicsFamilyIdx),
            static_cast<uint32_t>(g_vkCtx.presentFamilyIdx)
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
    vkCreateSwapchainKHR(g_vkCtx.device, &info, nullptr, &m_swapchain);

    // Save off swapchain details
    m_swapchainFormat = surfaceFormat.format;
    g_vkCtx.presentMode = presentMode;
    m_swapchainExtent = extent;

    // Retrieve the swapchain images from the device.
    // Note that VkImage is simply a handle like everything else.

    uint32_t numImages = 0;
    vkGetSwapchainImagesKHR(g_vkCtx.device, m_swapchain, &numImages, nullptr);
    if (numImages == 0)
    {
        throw std::runtime_error("vkGetSwapchainImagesKHR returned a zero image count.\n");
    }
    vkGetSwapchainImagesKHR(g_vkCtx.device, m_swapchain, &numImages, m_swapchainImages.data());
    if (numImages == 0)
    {
        throw std::runtime_error("vkGetSwapchainImagesKHR returned a zero image count.\n");
    }

    // Much like the logical device is an interface to the physical device,
    // image views are interfaces to actual images.  Think of it as this.
    // The image exists outside of you.  But the view is your personal view
    // ( how you perceive ) the image.
    for (uint32_t i = 0; i < FRAMES_IN_FLIGHT; i++)
    {
        m_swapchainViews[i] = CreateImageView(m_swapchainImages[i], m_swapchainFormat, VK_IMAGE_ASPECT_COLOR_BIT);
    }
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

    VkAttachmentDescription depthAttachment{};
    depthAttachment.format = FindDepthFormat();
    depthAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
    depthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    depthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    depthAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    depthAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    depthAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    depthAttachment.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

    VkAttachmentReference depthAttachmentRef{};
    depthAttachmentRef.attachment = 1;
    depthAttachmentRef.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

    VkSubpassDescription subpass{};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = 1;
    // The index of the attachment in this array is directly referenced from the fragment
    // shader with the "layout(location = 0) out vec4 outColor" directive
    subpass.pColorAttachments = &colorAttachmentRef;
    subpass.pDepthStencilAttachment = &depthAttachmentRef;

    VkSubpassDependency dependency{};
    dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
    dependency.srcStageMask =
        VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
    dependency.dstStageMask =
        VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
    dependency.dstAccessMask =
        VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

    const std::array attachments = {colorAttachment, depthAttachment};
    VkRenderPassCreateInfo renderPassInfo{};
    renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    renderPassInfo.attachmentCount = attachments.size();
    renderPassInfo.pAttachments = attachments.data();
    renderPassInfo.subpassCount = 1;
    renderPassInfo.pSubpasses = &subpass;
    renderPassInfo.dependencyCount = 1;
    renderPassInfo.pDependencies = &dependency;

    vkCreateRenderPass(g_vkCtx.device, &renderPassInfo, nullptr, &g_vkCtx.renderPass);
}

void RenderBackend::CreateDescriptorSetLayout()
{
    VkDescriptorSetLayoutBinding uboLayoutBinding{};
    uboLayoutBinding.binding = 0;
    uboLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    uboLayoutBinding.descriptorCount = 1;
    uboLayoutBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;

    VkDescriptorSetLayoutBinding samplerLayoutBinding{};
    samplerLayoutBinding.binding = 1;
    samplerLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    samplerLayoutBinding.descriptorCount = 1;
    samplerLayoutBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    const std::array bindings = {uboLayoutBinding, samplerLayoutBinding};

    VkDescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = bindings.size();
    layoutInfo.pBindings = bindings.data();

    if (vkCreateDescriptorSetLayout(g_vkCtx.device, &layoutInfo, nullptr, &m_descriptorSetLayout) != VK_SUCCESS)
    {
        throw std::runtime_error("Failed to create descriptor set layout!");
    }
}

void RenderBackend::CreateGraphicsPipeline()
{
    auto bindingDescription = Vertex_t::GetBindingDescription();
    auto attributeDescriptions = Vertex_t::GetAttributeDescriptions();

    // Vertex Input
    VkPipelineVertexInputStateCreateInfo vertexInputState{};
    vertexInputState.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vertexInputState.vertexBindingDescriptionCount = 1;
    vertexInputState.pVertexBindingDescriptions = &bindingDescription;
    vertexInputState.vertexAttributeDescriptionCount = attributeDescriptions.size();
    vertexInputState.pVertexAttributeDescriptions = attributeDescriptions.data();

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
    rasterizationState.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rasterizationState.depthBiasEnable = VK_FALSE;

    //Multisampling
    VkPipelineMultisampleStateCreateInfo multisampleState{};
    multisampleState.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisampleState.sampleShadingEnable = VK_FALSE;
    multisampleState.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
    multisampleState.minSampleShading = 1.0f; // Optional

    VkPipelineDepthStencilStateCreateInfo depthStencil{};
    depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    depthStencil.depthTestEnable = VK_TRUE;
    depthStencil.depthWriteEnable = VK_TRUE;
    depthStencil.depthCompareOp = VK_COMPARE_OP_LESS;
    depthStencil.depthBoundsTestEnable = VK_FALSE;
    depthStencil.minDepthBounds = 0.0f; // Optional
    depthStencil.maxDepthBounds = 1.0f; // Optional
    depthStencil.stencilTestEnable = VK_FALSE;

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
    pipelineLayoutInfo.setLayoutCount = 1;
    pipelineLayoutInfo.pSetLayouts = &m_descriptorSetLayout;

    vkCreatePipelineLayout(g_vkCtx.device, &pipelineLayoutInfo, nullptr, &m_pipelineLayout);

    VkGraphicsPipelineCreateInfo pipelineInfo{};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pipelineInfo.layout = m_pipelineLayout;
    pipelineInfo.renderPass = g_vkCtx.renderPass;
    pipelineInfo.subpass = 0;
    pipelineInfo.stageCount = shaderStages.size();
    pipelineInfo.pStages = shaderStages.data();
    pipelineInfo.pVertexInputState = &vertexInputState;
    pipelineInfo.pInputAssemblyState = &inputAssemblyState;
    pipelineInfo.pViewportState = &viewportState;
    pipelineInfo.pRasterizationState = &rasterizationState;
    pipelineInfo.pMultisampleState = &multisampleState;
    pipelineInfo.pDepthStencilState = &depthStencil;
    pipelineInfo.pColorBlendState = &colorBlendState;
    pipelineInfo.pDynamicState = &dynamicState;
    pipelineInfo.basePipelineHandle = VK_NULL_HANDLE; // Optional
    pipelineInfo.basePipelineIndex = -1; // Optional

    vkCreateGraphicsPipelines(g_vkCtx.device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &m_pipeline);

    vkDestroyShaderModule(g_vkCtx.device, vertShaderModule, nullptr);
    vkDestroyShaderModule(g_vkCtx.device, fragShaderModule, nullptr);
}

void RenderBackend::CreateFrameBuffers()
{
    for (size_t i = 0; i < m_swapchainImages.size(); i++) {
        const std::array attachments = {
            m_swapchainViews[i],
            m_depthImageView
        };

        VkFramebufferCreateInfo framebufferInfo{};
        framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        framebufferInfo.renderPass = g_vkCtx.renderPass;
        framebufferInfo.attachmentCount = attachments.size();
        framebufferInfo.pAttachments = attachments.data();
        framebufferInfo.width = m_swapchainExtent.width;
        framebufferInfo.height = m_swapchainExtent.height;
        framebufferInfo.layers = 1;

        vkCreateFramebuffer(g_vkCtx.device, &framebufferInfo, nullptr, &m_swapchainFramebuffers[i]);
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
    renderPassBeginInfo.renderPass = g_vkCtx.renderPass;
    renderPassBeginInfo.framebuffer = m_swapchainFramebuffers[imageIndex];
    renderPassBeginInfo.renderArea.offset = {0, 0};
    renderPassBeginInfo.renderArea.extent = m_swapchainExtent;

    constexpr VkClearValue clearColor = {{0.0f, 0.0f, 0.0f, 1.0f}};
    constexpr VkClearValue clearDepth = {1.0f, 0};
    constexpr std::array clearValues = {clearColor, clearDepth};

    renderPassBeginInfo.clearValueCount = clearValues.size();
    renderPassBeginInfo.pClearValues = clearValues.data();

    vkCmdBeginRenderPass(commandBuffer, &renderPassBeginInfo, VK_SUBPASS_CONTENTS_INLINE);

    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipeline);

    const VkBuffer vertexBuffers[] = {m_vertexBuffer};
    constexpr VkDeviceSize offsets[] = {0};
    vkCmdBindVertexBuffers(commandBuffer, 0, 1, vertexBuffers, offsets);

    vkCmdBindIndexBuffer(commandBuffer, m_indexBuffer, 0, VK_INDEX_TYPE_UINT32);

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

    vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipelineLayout, 0, 1,
        &m_descriptorSets[m_currentFrame], 0, nullptr);

    vkCmdDrawIndexed(commandBuffer, m_indices.size(), 1, 0, 0, 0);

    vkCmdEndRenderPass(commandBuffer);

    if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS)
    {
        throw std::runtime_error("Failed to record command buffer!");
    }
}

void RenderBackend::DrawFrame()
{
    vkWaitForFences(g_vkCtx.device, 1, &m_commandBufferFences[m_currentFrame], VK_TRUE, UINT64_MAX);

    uint32_t imageIndex;
    if (const VkResult result = vkAcquireNextImageKHR(
        g_vkCtx.device, m_swapchain, UINT64_MAX, m_acquireSemaphores[m_currentFrame], VK_NULL_HANDLE, &imageIndex);
        result == VK_ERROR_OUT_OF_DATE_KHR)
    {
        RecreateSwapchain();
        return;
    }
    else if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR)
    {
        throw std::runtime_error("Failed to acquire swap chain image!");
    }

    vkResetFences(g_vkCtx.device, 1, &m_commandBufferFences[m_currentFrame]);

    vkResetCommandBuffer(m_commandBuffers[m_currentFrame], 0);
    RecordCommandbuffer(m_commandBuffers[m_currentFrame], imageIndex);

    UpdateUniformBuffer(m_currentFrame);

    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;

    const VkSemaphore waitSemaphores[] = {m_acquireSemaphores[m_currentFrame]};
    constexpr VkPipelineStageFlags waitStages[] = {VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};
    submitInfo.waitSemaphoreCount = std::size(waitSemaphores);
    submitInfo.pWaitSemaphores = waitSemaphores;
    submitInfo.pWaitDstStageMask = waitStages;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &m_commandBuffers[m_currentFrame];

    const VkSemaphore signalSemaphores[] = {m_renderCompleteSemaphores[m_currentFrame]};
    submitInfo.signalSemaphoreCount = 1;
    submitInfo.pSignalSemaphores = signalSemaphores;

    if (vkQueueSubmit(g_vkCtx.graphicsQueue, 1, &submitInfo, m_commandBufferFences[m_currentFrame]) != VK_SUCCESS)
    {
        throw std::runtime_error("Failed to submit draw command buffer!");
    }

    VkPresentInfoKHR presentInfo{};
    presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;

    presentInfo.waitSemaphoreCount = std::size(signalSemaphores);
    presentInfo.pWaitSemaphores = signalSemaphores;

    const VkSwapchainKHR swapChains[] = {m_swapchain};
    presentInfo.swapchainCount = std::size(swapChains);
    presentInfo.pSwapchains = swapChains;
    presentInfo.pImageIndices = &imageIndex;

    if (const VkResult result = vkQueuePresentKHR(g_vkCtx.presentQueue, &presentInfo);
        result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR || m_frameBufferResized)
    {
        m_frameBufferResized = false;
        RecreateSwapchain();
    }
    else if (result != VK_SUCCESS)
    {
        throw std::runtime_error("Failed to present swap chain image!");
    }

    m_currentFrame = ++m_currentFrame % FRAMES_IN_FLIGHT;
}

void RenderBackend::RecreateSwapchain()
{
    // TODO: Research on the validation error that triggers each time resizing the window

    int width, height;
    glfwGetFramebufferSize(m_window, &width, &height);

    // After the window is minimized, wait until it is restored
    while (width == 0 || height == 0)
    {
        glfwGetFramebufferSize(m_window, &width, &height);
        glfwWaitEvents();
    }

    vkDeviceWaitIdle(g_vkCtx.device);

    CleanupSwapchain();

    CreateSwapChain();
    CreateDepthResources();
    CreateFrameBuffers();
}

void RenderBackend::CleanupSwapchain()
{
    vkDestroyImageView(g_vkCtx.device, m_depthImageView, nullptr);
    vkDestroyImage(g_vkCtx.device, m_depthImage, nullptr);
    vkFreeMemory(g_vkCtx.device, m_depthImageMemory, nullptr);

    for (const VkFramebuffer& frameBuffer : m_swapchainFramebuffers)
    {
        vkDestroyFramebuffer(g_vkCtx.device, frameBuffer, nullptr);
    }

    for (const VkImageView& imageView : m_swapchainViews)
    {
        vkDestroyImageView(g_vkCtx.device, imageView, nullptr);
    }

    vkDestroySwapchainKHR(g_vkCtx.device, m_swapchain, nullptr);
}

VkExtent2D RenderBackend::ChooseSurfaceExtent(const VkSurfaceCapabilitiesKHR& caps)
{
    VkExtent2D extent;

    int winWidth, winHeight;
    glfwGetWindowSize(m_window, &winWidth, &winHeight);

    // The extent is typically the size of the window we created the surface from.
    // However if Vulkan returns -1 then simply substitute the window size.
    if (caps.currentExtent.width == -1 ||
        caps.currentExtent.width != winWidth ||
        caps.currentExtent.height != winHeight)
    {
        extent.width = winWidth;
        extent.height = winHeight;
    }
    else
    {
        extent = caps.currentExtent;
    }

    return extent;
}

void RenderBackend::UpdateUniformBuffer(const uint32_t currentFrame)
{
    static auto startTime = std::chrono::high_resolution_clock::now();

    const auto currentTime = std::chrono::high_resolution_clock::now();
    const float time = std::chrono::duration<float, std::chrono::seconds::period>(currentTime - startTime).count();

    UniformBufferObject_t ubo{};
    ubo.model = glm::rotate(glm::mat4(1.0f), time * glm::radians(90.0f), glm::vec3(0.0f, 0.0f, 1.0f));
    ubo.view = glm::lookAt(glm::vec3(2.0f, 2.0f, 2.0f), glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 0.0f, 1.0f));
    ubo.proj = glm::perspective(glm::radians(45.0f),
        static_cast<float>(m_swapchainExtent.width) / static_cast<float>(m_swapchainExtent.height), 0.1f, 10.0f);
    // GLM was originally designed for OpenGL, where the Y coordinate of the clip coordinates is inverted. The easiest
    // way to compensate for that is to flip the sign on the scaling factor of the Y axis in the projection matrix.
    ubo.proj[1][1] *= -1;

    memcpy(m_uniformBufferMappings[currentFrame], &ubo, sizeof(ubo));
}

void RenderBackend::CopyBuffer(VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size, VkCommandPool commandPool)
{
    VkCommandBuffer commandBuffer = BeginSingleTimeCommands();

    VkBufferCopy copyRegion{};
    copyRegion.size = size;
    vkCmdCopyBuffer(commandBuffer, srcBuffer, dstBuffer, 1, &copyRegion);

    EndSingleTimeCommands(commandBuffer);
}

VkCommandBuffer RenderBackend::BeginSingleTimeCommands()
{
    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandPool = m_commandPool;
    allocInfo.commandBufferCount = 1;

    VkCommandBuffer commandBuffer;
    vkAllocateCommandBuffers(g_vkCtx.device, &allocInfo, &commandBuffer);

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(commandBuffer, &beginInfo);

    return commandBuffer;
}

void RenderBackend::EndSingleTimeCommands(const VkCommandBuffer& commandBuffer)
{
    vkEndCommandBuffer(commandBuffer);

    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer;

    vkQueueSubmit(g_vkCtx.graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE);
    vkQueueWaitIdle(g_vkCtx.graphicsQueue);

    vkFreeCommandBuffers(g_vkCtx.device, m_commandPool, 1, &commandBuffer);
}

void RenderBackend::TransitionImageLayout(
    VkImage image,
    VkFormat format,
    VkImageLayout oldLayout,
    VkImageLayout newLayout)
{
    VkCommandBuffer commandBuffer = BeginSingleTimeCommands();

    VkImageMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.oldLayout = oldLayout;
    barrier.newLayout = newLayout;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = image;
    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    barrier.subresourceRange.baseMipLevel = 0;
    barrier.subresourceRange.levelCount = 1;
    barrier.subresourceRange.baseArrayLayer = 0;
    barrier.subresourceRange.layerCount = 1;

    VkPipelineStageFlags sourceStage;
    VkPipelineStageFlags destinationStage;

    if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED && newLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL)
    {
        barrier.srcAccessMask = 0;
        barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;

        sourceStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
        destinationStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
    }
    else if (oldLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL && newLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
    {
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

        sourceStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
        destinationStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    }
    else
    {
        throw std::invalid_argument("Unsupported layout transition!");
    }

    vkCmdPipelineBarrier(
        commandBuffer,
        sourceStage, destinationStage,
        0,
        0, nullptr,
        0, nullptr,
        1, &barrier);

    EndSingleTimeCommands(commandBuffer);
}

void RenderBackend::CopyBufferToImage(VkBuffer buffer, VkImage image, uint32_t width, uint32_t height)
{
    VkCommandBuffer commandBuffer = BeginSingleTimeCommands();

    VkBufferImageCopy region{};
    region.bufferOffset = 0;
    region.bufferRowLength = 0;
    region.bufferImageHeight = 0;

    region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.imageSubresource.mipLevel = 0;
    region.imageSubresource.baseArrayLayer = 0;
    region.imageSubresource.layerCount = 1;

    region.imageOffset = {0, 0, 0};
    region.imageExtent = {
        .width = width,
        .height = height,
        .depth = 1
    };

    vkCmdCopyBufferToImage(
        commandBuffer,
        buffer,
        image,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        1,
        &region);

    EndSingleTimeCommands(commandBuffer);
}

VkImageView RenderBackend::CreateImageView(const VkImage& image, const VkFormat& format, VkImageAspectFlags aspectFlags)
{
    VkImageViewCreateInfo viewInfo{};
    viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewInfo.image = image;
    viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    viewInfo.format = format;

    // We don't need to swizzle (swap around) any of the color channels
    viewInfo.components.r = VK_COMPONENT_SWIZZLE_R;
    viewInfo.components.g = VK_COMPONENT_SWIZZLE_G;
    viewInfo.components.b = VK_COMPONENT_SWIZZLE_B;
    viewInfo.components.a = VK_COMPONENT_SWIZZLE_A;

    // The subresourceRange field describes what the image's purpose is and which part of
    // the image should be accessed
    viewInfo.subresourceRange.aspectMask = aspectFlags;
    viewInfo.subresourceRange.baseMipLevel = 0;
    viewInfo.subresourceRange.levelCount = 1;
    viewInfo.subresourceRange.baseArrayLayer = 0;
    viewInfo.subresourceRange.layerCount = 1;

    VkImageView imageView{};
    if (vkCreateImageView(g_vkCtx.device, &viewInfo, nullptr, &imageView) != VK_SUCCESS)
    {
        throw std::runtime_error("Failed to create texture image view!");
    }

    return imageView;
}

void RenderBackend::CreateTextureSampler()
{
    VkSamplerCreateInfo samplerInfo{};
    samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    samplerInfo.magFilter = VK_FILTER_LINEAR;
    samplerInfo.minFilter = VK_FILTER_LINEAR;
    samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;

    samplerInfo.anisotropyEnable = VK_TRUE;
    // A lower value = more performance but lower quality. To find the supported value
    // we are retriving physical device properties.
    samplerInfo.maxAnisotropy = g_vkCtx.gpu.props.limits.maxSamplerAnisotropy;

    // Which color is returned when sampling beyond the image
    samplerInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
    samplerInfo.unnormalizedCoordinates = VK_FALSE;
    samplerInfo.compareEnable = VK_FALSE;
    samplerInfo.compareOp = VK_COMPARE_OP_ALWAYS;

    samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    samplerInfo.mipLodBias = 0.0f;
    samplerInfo.minLod = 0.0f;
    samplerInfo.maxLod = 0.0f;

    if (vkCreateSampler(g_vkCtx.device, &samplerInfo, nullptr, &m_textureSampler) != VK_SUCCESS)
    {
        throw std::runtime_error("Failed to create texture sampler!");
    }
}
