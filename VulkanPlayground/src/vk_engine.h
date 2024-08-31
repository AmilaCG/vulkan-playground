#pragma once

#include "vk_types.h"

class VulkanEngine
{
public:
    static VulkanEngine& Get();

    //initializes everything in the engine
    void init();
    //shuts down the engine
    void cleanup();
    //draw loop
    void draw();
    //run main loop
    void run();

private:
    void init_vulkan();
    void init_swapchain();
    void init_commands();
    void init_sync_structures();

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
};
