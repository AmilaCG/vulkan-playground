#pragma once

#define VK_USE_PLATFORM_WIN32_KHR
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#define GLFW_EXPOSE_NATIVE_WIN32
#include <GLFW/glfw3native.h>

#include <vector>
#include <array>
#include <vulkan/vulkan_core.h>
#include <glm/glm.hpp>

// Everything that is needed by the backend needs to be double buffered to allow it to run in
// parallel on a dual cpu machine
static constexpr uint32_t FRAMES_IN_FLIGHT = 2;

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

    VkInstance                  instance{};
    VkDevice                    device{};
    int                         graphicsFamilyIdx{};
    int                         presentFamilyIdx{};
    VkQueue                     graphicsQueue{};
    VkQueue                     presentQueue{};
    VkPresentModeKHR            presentMode{};
    VkSurfaceKHR                surface{};

    VkFormat                    depthFormat{};
    VkRenderPass                renderPass{};

    bool                        enableValidation{true};
};

struct Vertex_t
{
    glm::vec3 pos;
    glm::vec3 color;
    glm::vec2 texCoord;

    static VkVertexInputBindingDescription GetBindingDescription()
    {
        VkVertexInputBindingDescription bindingDescription{};
        bindingDescription.binding = 0;
        bindingDescription.stride = sizeof(Vertex_t);
        bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

        return bindingDescription;
    }

    static std::array<VkVertexInputAttributeDescription, 3> GetAttributeDescriptions()
    {
        std::array<VkVertexInputAttributeDescription, 3> attributeDescriptions{};
        attributeDescriptions[0].binding = 0; // Location directive of the input in vertex shader
        attributeDescriptions[0].location = 0;
        attributeDescriptions[0].format = VK_FORMAT_R32G32B32_SFLOAT;
        attributeDescriptions[0].offset = offsetof(Vertex_t, pos);

        attributeDescriptions[1].binding = 0;
        attributeDescriptions[1].location = 1;
        attributeDescriptions[1].format = VK_FORMAT_R32G32B32_SFLOAT;
        attributeDescriptions[1].offset = offsetof(Vertex_t, color);

        attributeDescriptions[2].binding = 0;
        attributeDescriptions[2].location = 2;
        attributeDescriptions[2].format = VK_FORMAT_R32G32_SFLOAT;
        attributeDescriptions[2].offset = offsetof(Vertex_t, texCoord);

        return attributeDescriptions;
    }
};

struct UniformBufferObject_t {
    alignas(16) glm::mat4 model;
    alignas(16) glm::mat4 view;
    alignas(16) glm::mat4 proj;
};

class RenderBackend
{
public:
    RenderBackend();
    ~RenderBackend();

    void Init();
    void Shutdown();
    void RunRenderLoop();
    void SetFramebufferResizeFlag(bool resized);

private:
    bool WindowInit();
    void CreateInstance();
    void CreateSurface();
    void SelectSuitablePhysicalDevice();
    void CreateLogicalDeviceAndQueues();
    void CreateSemaphores();
    void CreateCommandPool();
    void CreateTextureImage();
    void CreateTextureImageView();
    void CreateVertexBuffer();
    void CreateIndexBuffer();
    void CreateUniformBuffers();
    void CreateDescriptorPool();
    void CreateDescriptorSets();
    void CreateCommandBuffers();
    void CreateSwapChain();
    void CreateRenderPass();
    void CreateDescriptorSetLayout();
    void CreateGraphicsPipeline();
    void CreateFrameBuffers();
    void RecordCommandbuffer(const VkCommandBuffer& commandBuffer, uint32_t imageIndex);
    void DrawFrame();
    void RecreateSwapchain();
    void CleanupSwapchain();

    VkExtent2D ChooseSurfaceExtent(const VkSurfaceCapabilitiesKHR& caps);
    void UpdateUniformBuffer(uint32_t currentFrame);
    void CopyBuffer(VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size, VkCommandPool commandPool);
    VkCommandBuffer BeginSingleTimeCommands();
    void EndSingleTimeCommands(const VkCommandBuffer& commandBuffer);
    void TransitionImageLayout(VkImage image, VkFormat format, VkImageLayout oldLayout, VkImageLayout newLayout);
    void CopyBufferToImage(VkBuffer buffer, VkImage image, uint32_t width, uint32_t height);
    VkImageView CreateImageView(const VkImage& image, const VkFormat& format);
    void CreateTextureSampler();

private:
    GLFWwindow*                     m_window{nullptr};

    uint32_t                        m_currentFrame{0};

    std::vector<const char*>        m_instanceExtensions;
    std::vector<const char*>        m_deviceExtensions;
    std::vector<const char*>        m_validationLayers;

    std::vector<VkSemaphore>        m_acquireSemaphores{FRAMES_IN_FLIGHT};
    std::vector<VkSemaphore>        m_renderCompleteSemaphores{FRAMES_IN_FLIGHT};
    std::vector<VkFence>            m_commandBufferFences{FRAMES_IN_FLIGHT};

    VkCommandPool                   m_commandPool{};
    std::vector<VkCommandBuffer>    m_commandBuffers{FRAMES_IN_FLIGHT};
    VkSwapchainKHR                  m_swapchain{};
    VkFormat                        m_swapchainFormat{};
    VkExtent2D                      m_swapchainExtent{};
    std::vector<VkImage>            m_swapchainImages{FRAMES_IN_FLIGHT};
    std::vector<VkImageView>        m_swapchainViews{FRAMES_IN_FLIGHT};
    std::vector<VkFramebuffer>      m_swapchainFramebuffers{FRAMES_IN_FLIGHT};
    bool                            m_frameBufferResized{false};

    VkDescriptorPool                m_descriptorPool{};
    std::vector<VkDescriptorSet>    m_descriptorSets{FRAMES_IN_FLIGHT};
    VkDescriptorSetLayout           m_descriptorSetLayout{};
    VkPipelineLayout                m_pipelineLayout{};
    VkPipeline                      m_pipeline{};

    VkBuffer                        m_vertexBuffer{};
    VkDeviceMemory                  m_vertexBufferMemory{};
    VkBuffer                        m_indexBuffer{};
    VkDeviceMemory                  m_indexBufferMemory{};
    VkImage                         m_textureImage{};
    VkImageView                     m_textureImageView{};
    VkDeviceMemory                  m_textureImageMemory{};
    VkSampler                       m_textureSampler{};

    std::vector<VkBuffer>           m_uniformBuffers{FRAMES_IN_FLIGHT};
    std::vector<VkDeviceMemory>     m_uniformBufferMemories{FRAMES_IN_FLIGHT};
    std::vector<void*>              m_uniformBufferMappings{FRAMES_IN_FLIGHT};
};
