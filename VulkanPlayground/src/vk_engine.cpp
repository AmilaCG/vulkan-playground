#include "vk_engine.h"

#include <chrono>
#include <thread>

#include <SDL.h>
#include <SDL_vulkan.h>
#include <VkBootstrap.h>

#include "vk_initializers.h"
#include "vk_types.h"

VulkanEngine* loadedEngine = nullptr;
constexpr bool bUseValidationLayers = true;

VulkanEngine& VulkanEngine::Get() { return *loadedEngine; }

void VulkanEngine::init()
{
    // only one engine initialization is allowed with the application.
    assert(loadedEngine == nullptr);
    loadedEngine = this;

    // We initialize SDL and create a window with it.
    SDL_Init(SDL_INIT_VIDEO);

    constexpr SDL_WindowFlags window_flags = SDL_WINDOW_VULKAN;

    _window = SDL_CreateWindow(
        "Vulkan Engine",
        SDL_WINDOWPOS_UNDEFINED,
        SDL_WINDOWPOS_UNDEFINED,
        _windowExtent.width,
        _windowExtent.height,
        window_flags);
    fmt::print("{}\n", SDL_GetError());

    init_vulkan();
    init_swapchain();
    init_commands();
    init_sync_structures();

    // everything went fine
    _isInitialized = true;
}

void VulkanEngine::cleanup()
{
    if (_isInitialized)
    {
        SDL_DestroyWindow(_window);
    }

    // clear engine pointer
    loadedEngine = nullptr;
}

void VulkanEngine::draw()
{
    // nothing yet
}

void VulkanEngine::run()
{
    SDL_Event e;
    bool bQuit = false;

    // main loop
    while (!bQuit)
    {
        // Handle events on queue
        while (SDL_PollEvent(&e) != 0)
        {
            // close the window when user alt-f4s or clicks the X button
            if (e.type == SDL_QUIT)
                bQuit = true;

            if (e.type == SDL_WINDOWEVENT)
            {
                if (e.window.event == SDL_WINDOWEVENT_MINIMIZED)
                {
                    _stopRendering = true;
                }
                if (e.window.event == SDL_WINDOWEVENT_RESTORED)
                {
                    _stopRendering = false;
                }
            }
        }

        // do not draw if we are minimized
        if (_stopRendering)
        {
            // throttle the speed to avoid the endless spinning
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            continue;
        }

        draw();
    }
}

void VulkanEngine::init_vulkan()
{
    vkb::InstanceBuilder builder;
    // Make the vulkan instance with basic debug features
    vkb::Instance vkb_inst = builder.set_app_name("VK Guide")
        .request_validation_layers(bUseValidationLayers)
        .use_default_debug_messenger()
        .require_api_version(1, 3, 0)
        .build()
        .value();

    _instance = vkb_inst.instance;
    _debug_messenger = vkb_inst.debug_messenger;

    SDL_Vulkan_CreateSurface(_window, _instance, &_surface);

    // Vulkan 1.3 features
    VkPhysicalDeviceVulkan13Features features13{};
    features13.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
    features13.synchronization2 = true;
    features13.dynamicRendering = true;

    // Vulkan 1.2 features
    VkPhysicalDeviceVulkan12Features features12{};
    features12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    features12.descriptorIndexing = true;
    features12.bufferDeviceAddress = true;

    // Selecting a GPU that can write to the SDL surface and supports Vulkan 1.3
    // with correct features
    vkb::PhysicalDeviceSelector selector{vkb_inst};
    vkb::PhysicalDevice physicalDevice = selector
        .set_minimum_version(1, 3)
        .set_required_features_13(features13)
        .set_required_features_12(features12)
        .set_surface(_surface)
        .select()
        .value();

    // Create the final vulkan device
    vkb::DeviceBuilder deviceBuilder{physicalDevice};
    vkb::Device vkbDevice = deviceBuilder.build().value();

    _device = vkbDevice.device;
    _chosenGPU = physicalDevice.physical_device;
}

void VulkanEngine::init_swapchain()
{
}

void VulkanEngine::init_commands()
{
}

void VulkanEngine::init_sync_structures()
{
}
