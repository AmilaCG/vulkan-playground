#include "vk_engine.h"

#include <chrono>
#include <thread>

#include <SDL.h>
#include <SDL_vulkan.h>
#include <VkBootstrap.h>
#include <imgui.h>
#include <imgui_impl_sdl2.h>
#include <imgui_impl_vulkan.h>
#define VMA_IMPLEMENTATION
#include <vk_mem_alloc.h>

#include "vk_initializers.h"
#include "vk_types.h"
#include "vk_images.h"
#include "vk_pipelines.h"

auto COMP_SHADER_PATH_GRADIENT = "gradient_color.comp.spv";
auto COMP_SHADER_PATH_SKY = "sky.comp.spv";
auto FRAG_SHADER_TRIANGLE = "colored_tiangle.frag.spv";
auto VERT_SHADER_TRIANGLE = "colored_triangle.vert.spv";

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
    init_descriptors();
    init_pipelines();
    init_imgui();

    // everything went fine
    _isInitialized = true;
}

void VulkanEngine::cleanup()
{
    if (_isInitialized)
    {
        vkDeviceWaitIdle(_device);

        for (FrameData& frame : _frames)
        {
            vkDestroyCommandPool(_device, frame._commandPool, nullptr);

            vkDestroyFence(_device, frame._renderFence, nullptr);
            vkDestroySemaphore(_device, frame._renderSemaphore, nullptr);
            vkDestroySemaphore(_device, frame._swapchainSemaphore, nullptr);

            frame._deletionQueue.flush();
        }

        _mainDeletionQueue.flush();

        destroy_swapchain();

        vkDestroySurfaceKHR(_instance, _surface, nullptr);
        vkDestroyDevice(_device, nullptr);

        vkb::destroy_debug_utils_messenger(_instance, _debug_messenger);
        vkDestroyInstance(_instance, nullptr);
        SDL_DestroyWindow(_window);
    }

    // clear engine pointer
    loadedEngine = nullptr;
}

void VulkanEngine::draw()
{
    FrameData currentFrame = get_current_frame();

    // Wait until the GPU has finished rendering the last frame
    VK_CHECK(vkWaitForFences(_device, 1, &currentFrame._renderFence, true, ONE_SEC_NS));

    currentFrame._deletionQueue.flush();

    VK_CHECK(vkResetFences(_device, 1, &currentFrame._renderFence));

    uint32_t swapchainImageIndex;
    // Request image from the swapchain
    VK_CHECK(vkAcquireNextImageKHR(_device,
                           _swapchain,
                           ONE_SEC_NS,
                           currentFrame._swapchainSemaphore,
                           nullptr,
                           &swapchainImageIndex));

    VkCommandBuffer cmd = currentFrame._mainCommandBuffer;

    // Now that we are sure that the commands finished executing, we can safely reset
    // the command buffer to begin recording again
    VK_CHECK(vkResetCommandBuffer(cmd, 0));

    // Begin the command buffer recording. We will be using this command buffer exactly once.
    VkCommandBufferBeginInfo cmdBeginInfo =
        vkinit::command_buffer_begin_info(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT /* Optional flag */);

    VK_CHECK(vkBeginCommandBuffer(cmd, &cmdBeginInfo));

    _drawExtent.width = _drawImage.imageExtent.width;
    _drawExtent.height = _drawImage.imageExtent.height;

    // Make the draw image into general layout so we can write on it. As we overwrite it, we don't
    // care about its old layout
    vkutil::transition_image(cmd,
                             _drawImage.image,
                             VK_IMAGE_LAYOUT_UNDEFINED,
                             VK_IMAGE_LAYOUT_GENERAL);

    draw_background(cmd);

    // Transition draw image into transfer source layout since we are going to copy it to the swapchain
    vkutil::transition_image(cmd,
                             _drawImage.image,
                             VK_IMAGE_LAYOUT_GENERAL,
                             VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);

    // Transfer swapchaing image into transfer dst layout to copy incoming draw image into this
    vkutil::transition_image(cmd,
                             _swapchainImages[swapchainImageIndex],
                             VK_IMAGE_LAYOUT_UNDEFINED,
                             VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);

    // Copy the draw image into the swapchain
    vkutil::copy_image_to_image(cmd,
                                _drawImage.image,
                                _swapchainImages[swapchainImageIndex],
                                _drawExtent,
                                _swapchainExtent);

    // Draw imgui UI on the swapchain image
    draw_imgui(cmd, _swapchainImageViews[swapchainImageIndex]);

    // Set swapchain image layout to present so we can show it on the screen
    vkutil::transition_image(cmd,
                             _swapchainImages[swapchainImageIndex],
                             VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                             VK_IMAGE_LAYOUT_PRESENT_SRC_KHR);

    // Finalize the command buffer
    VK_CHECK(vkEndCommandBuffer(cmd));

    VkCommandBufferSubmitInfo cmdInfo = vkinit::command_buffer_submit_info(cmd);

    // Wait on the swapchain (present) semaphore, as that is signaled when the swapchain is ready
    VkSemaphoreSubmitInfo waitInfo = vkinit::semaphore_submit_info(
        VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT_KHR, currentFrame._swapchainSemaphore);

    // Signal the render semaphore when rendering is finished
    VkSemaphoreSubmitInfo signalInfo = vkinit::semaphore_submit_info(
        VK_PIPELINE_STAGE_2_ALL_GRAPHICS_BIT, currentFrame._renderSemaphore);

    VkSubmitInfo2 submit = vkinit::submit_info(&cmdInfo, &signalInfo, &waitInfo);

    // Submit command buffer to the queue to execute it
    VK_CHECK(vkQueueSubmit2(_graphicsQueue, 1, &submit, currentFrame._renderFence));

    // Prepare to present (show rendered image in the window)
    VkPresentInfoKHR presentInfo{};
    presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    presentInfo.pSwapchains = &_swapchain;
    presentInfo.swapchainCount = 1;

    // We want to wait on the render semaphore for presenting
    presentInfo.pWaitSemaphores = &currentFrame._renderSemaphore;
    presentInfo.waitSemaphoreCount = 1;

    presentInfo.pImageIndices = &swapchainImageIndex;

    VK_CHECK(vkQueuePresentKHR(_graphicsQueue, &presentInfo));

    _frameNumber++;
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

            // Send SDL event to imgui for handling
            ImGui_ImplSDL2_ProcessEvent(&e);
        }

        // do not draw if we are minimized
        if (_stopRendering)
        {
            // throttle the speed to avoid the endless spinning
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            continue;
        }

        // Imgui new frame
        ImGui_ImplVulkan_NewFrame();
        ImGui_ImplSDL2_NewFrame();
        ImGui::NewFrame();

        if (ImGui::Begin("background"))
        {
            ComputeEffect& selected = backgroundEffects[currentBackgroundEffect];

            ImGui::Text("Selected effect: ", selected.name);

            ImGui::SliderInt("Effect Index", &currentBackgroundEffect, 0, backgroundEffects.size() - 1);

            ImGui::InputFloat4("data1", (float*)& selected.data.data1);
            ImGui::InputFloat4("data2", (float*)& selected.data.data2);
            ImGui::InputFloat4("data3", (float*)& selected.data.data3);
            ImGui::InputFloat4("data4", (float*)& selected.data.data4);
        }
        ImGui::End();

        // Make imgui calculate it's internal draw structures. It will not draw anything yet.
        ImGui::Render();

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

    _graphicsQueue = vkbDevice.get_queue(vkb::QueueType::graphics).value();
    _graphicsQueueFamily = vkbDevice.get_queue_index(vkb::QueueType::graphics).value();

    // Create a command pool
    VkCommandPoolCreateInfo commandPoolInfo =
        vkinit::command_pool_create_info(_graphicsQueueFamily, VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT);

    for (FrameData& frame : _frames)
    {
        VK_CHECK(vkCreateCommandPool(_device, &commandPoolInfo, nullptr, &frame._commandPool));

        // Allocate default command buffer that used for rendering
        VkCommandBufferAllocateInfo cmdAllocInfo = vkinit::command_buffer_allocate_info(frame._commandPool);

        VK_CHECK(vkAllocateCommandBuffers(_device, &cmdAllocInfo, &frame._mainCommandBuffer));
    }

    // Initialize the memory allocator
    VmaAllocatorCreateInfo allocatorInfo{};
    allocatorInfo.physicalDevice = _chosenGPU;
    allocatorInfo.device = _device;
    allocatorInfo.instance = _instance;
    allocatorInfo.flags = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT;
    vmaCreateAllocator(&allocatorInfo, &_allocator);

    _mainDeletionQueue.push_function([&]()
    {
        vmaDestroyAllocator(_allocator);
    });
}

void VulkanEngine::init_swapchain()
{
    create_swapchain(_windowExtent.width, _windowExtent.height);

    const VkExtent3D drawImageExtent = {
        _windowExtent.width,
        _windowExtent.height,
        1
    };

    _drawImage.imageFormat = VK_FORMAT_R16G16B16A16_SFLOAT;
    _drawImage.imageExtent = drawImageExtent;

    VkImageUsageFlags drawImageUsages{};
    drawImageUsages |= VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    drawImageUsages |= VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    drawImageUsages |= VK_IMAGE_USAGE_STORAGE_BIT;
    drawImageUsages |= VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

    VkImageCreateInfo rImgInfo = vkinit::image_create_info(_drawImage.imageFormat, drawImageUsages, drawImageExtent);

    // Allocate GPU memory for the draw image
    VmaAllocationCreateInfo rImgAllocInfo{};
    rImgAllocInfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;
    rImgAllocInfo.requiredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

    // Allocate and create the image
    vmaCreateImage(_allocator, &rImgInfo, &rImgAllocInfo, &_drawImage.image, &_drawImage.allocation, nullptr);

    // Build an image-view for the draw image
    const VkImageViewCreateInfo rViewInfo = vkinit::imageview_create_info(_drawImage.imageFormat,
                                                                    _drawImage.image,
                                                                    VK_IMAGE_ASPECT_COLOR_BIT);

    VK_CHECK(vkCreateImageView(_device, &rViewInfo, nullptr, &_drawImage.imageView));

    _mainDeletionQueue.push_function([=]()
    {
        vkDestroyImageView(_device, _drawImage.imageView, nullptr);
        vmaDestroyImage(_allocator, _drawImage.image, _drawImage.allocation);
    });
}

void VulkanEngine::init_commands()
{
    VkCommandPoolCreateInfo commandPoolInfo =
        vkinit::command_pool_create_info(_graphicsQueueFamily, VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT);

    VK_CHECK(vkCreateCommandPool(_device, &commandPoolInfo, nullptr, &_immCommandPool));

    VkCommandBufferAllocateInfo cmdAllocInfo = vkinit::command_buffer_allocate_info(_immCommandPool, 1);

    VK_CHECK(vkAllocateCommandBuffers(_device, &cmdAllocInfo, &_immCommandBuffer));

    _mainDeletionQueue.push_function([=]()
    {
        vkDestroyCommandPool(_device, _immCommandPool, nullptr);
    });
}

void VulkanEngine::init_sync_structures()
{
    VkFenceCreateInfo fenceCreateInfo = vkinit::fence_create_info(VK_FENCE_CREATE_SIGNALED_BIT);
    VkSemaphoreCreateInfo semaphoreCreateInfo = vkinit::semaphore_create_info();

    for (FrameData& frame : _frames)
    {
        VK_CHECK(vkCreateFence(_device, &fenceCreateInfo, nullptr, &frame._renderFence));

        VK_CHECK(vkCreateSemaphore(_device, &semaphoreCreateInfo, nullptr, &frame._swapchainSemaphore));
        VK_CHECK(vkCreateSemaphore(_device, &semaphoreCreateInfo, nullptr, &frame._renderSemaphore));
    }

    VK_CHECK(vkCreateFence(_device, &fenceCreateInfo, nullptr, &_immFence));
    _mainDeletionQueue.push_function([=]()
    {
        vkDestroyFence(_device, _immFence, nullptr);
    });
}

void VulkanEngine::create_swapchain(uint32_t width, uint32_t height)
{
    _swapchainImageFormat = VK_FORMAT_R8G8B8A8_UNORM;

    vkb::SwapchainBuilder swapchainBuilder{_chosenGPU, _device, _surface};

    VkSurfaceFormatKHR surfaceFormat{};
    surfaceFormat.format = _swapchainImageFormat;
    surfaceFormat.colorSpace = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;

    vkb::Swapchain vkbSwapchain = swapchainBuilder
        .set_desired_format(surfaceFormat)
        .set_desired_present_mode(VK_PRESENT_MODE_FIFO_KHR)
        .set_desired_extent(width, height)
        .add_image_usage_flags(VK_IMAGE_USAGE_TRANSFER_DST_BIT)
        .build()
        .value();

    _swapchainExtent = vkbSwapchain.extent;
    _swapchain = vkbSwapchain.swapchain;
    _swapchainImages = vkbSwapchain.get_images().value();
    _swapchainImageViews = vkbSwapchain.get_image_views().value();
}

void VulkanEngine::destroy_swapchain()
{
    vkDestroySwapchainKHR(_device, _swapchain, nullptr);

    for (const auto& imageView : _swapchainImageViews)
    {
        vkDestroyImageView(_device, imageView, nullptr);
    }
}

FrameData& VulkanEngine::get_current_frame()
{
    return _frames[_frameNumber % FRAME_OVERLAP];
}

void VulkanEngine::draw_background(VkCommandBuffer cmd)
{
    ComputeEffect& effect = backgroundEffects[currentBackgroundEffect];

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, effect.pipeline);

    vkCmdBindDescriptorSets(cmd,
                            VK_PIPELINE_BIND_POINT_COMPUTE,
                            _gradientPipelineLayout,
                            0,
                            1,
                            &_drawImageDescriptors,
                            0,
                            nullptr);

    vkCmdPushConstants(cmd,
                       _gradientPipelineLayout,
                       VK_SHADER_STAGE_COMPUTE_BIT,
                       0,
                       sizeof(ComputePushConstants),
                       &effect.data);

    // Execute the compute pipeline dispatch. We are using 16x16 workgroup size so we
    // need to divide by it
    vkCmdDispatch(cmd, std::ceil(_drawExtent.width / 16.0), std::ceil(_drawExtent.height / 16.0), 1);
}

void VulkanEngine::init_descriptors()
{
    std::vector<DescriptorAllocator::PoolSizeRatio> sizes = {
        {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1}
    };

    // Create a descriptor pool that will hold 10 sets with 1 image each
    _globalDescriptorAllocator.init_pool(_device, 10, sizes);

    DescriptorLayoutBuilder builder;
    builder.add_binding(0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);
    // Create a layout with a single VK_DESCRIPTOR_TYPE_STORAGE_IMAGE binding at binding 0
    _drawImageDescriptorLayout = builder.build(_device, VK_SHADER_STAGE_COMPUTE_BIT);

    // Allocate a descriptor set for the draw image
    _drawImageDescriptors = _globalDescriptorAllocator.allocate(_device, _drawImageDescriptorLayout);

    VkDescriptorImageInfo imageInfo{};
    imageInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    imageInfo.imageView = _drawImage.imageView;

    VkWriteDescriptorSet drawImageWrite{};
    drawImageWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    drawImageWrite.dstBinding = 0;
    drawImageWrite.dstSet = _drawImageDescriptors;
    drawImageWrite.descriptorCount = 1;
    drawImageWrite.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    drawImageWrite.pImageInfo = &imageInfo;

    vkUpdateDescriptorSets(_device, 1, &drawImageWrite, 0, nullptr);

    _mainDeletionQueue.push_function([&]()
    {
        _globalDescriptorAllocator.destroy_pool(_device);
        vkDestroyDescriptorSetLayout(_device, _drawImageDescriptorLayout, nullptr);
    });
}

void VulkanEngine::init_pipelines()
{
    init_background_pipelines();
    init_triangle_pipeline();
}

void VulkanEngine::init_background_pipelines()
{
    VkPipelineLayoutCreateInfo computeLayout{};
    computeLayout.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    computeLayout.pSetLayouts = &_drawImageDescriptorLayout;
    computeLayout.setLayoutCount = 1;

    VkPushConstantRange pushConstantRange{};
    pushConstantRange.offset = 0;
    pushConstantRange.size = sizeof(ComputePushConstants);
    pushConstantRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    computeLayout.pPushConstantRanges = &pushConstantRange;
    computeLayout.pushConstantRangeCount = 1;

    VK_CHECK(vkCreatePipelineLayout(_device, &computeLayout, nullptr, &_gradientPipelineLayout));

    VkShaderModule gradientShader;
    if (!vkutil::load_shader_module(COMP_SHADER_PATH_GRADIENT, _device, gradientShader))
    {
        fmt::print("Compute shader creation failed!\n");
    }

    VkShaderModule skyShader;
    if (!vkutil::load_shader_module(COMP_SHADER_PATH_SKY, _device, skyShader))
    {
        fmt::print("Compute shader creation failed!\n");
    }

    VkPipelineShaderStageCreateInfo stageInfo{};
    stageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stageInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    stageInfo.module = gradientShader;
    stageInfo.pName = "main";

    VkComputePipelineCreateInfo computePipelineCreateInfo{};
    computePipelineCreateInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    computePipelineCreateInfo.layout = _gradientPipelineLayout;
    computePipelineCreateInfo.stage = stageInfo;

    ComputeEffect gradient{};
    gradient.layout = _gradientPipelineLayout;
    gradient.name = "gradient";
    gradient.data = {};
    // Default colors
    gradient.data.data1 = glm::vec4(1, 0, 0, 1);
    gradient.data.data2 = glm::vec4(0, 0, 1, 1);

    VK_CHECK(vkCreateComputePipelines(_device,
                                      VK_NULL_HANDLE,
                                      1,
                                      &computePipelineCreateInfo,
                                      nullptr,
                                      &gradient.pipeline));

    // Change the shader module only to create the sky shader
    computePipelineCreateInfo.stage.module = skyShader;

    ComputeEffect sky{};
    sky.layout = _gradientPipelineLayout;
    sky.name = "sky";
    sky.data = {};
    // Default sky parameters
    sky.data.data1 = glm::vec4(0.1, 0.2, 0.4, 0.97);

    VK_CHECK(vkCreateComputePipelines(_device,
                                      VK_NULL_HANDLE,
                                      1,
                                      &computePipelineCreateInfo,
                                      nullptr,
                                      &sky.pipeline));

    backgroundEffects.push_back(gradient);
    backgroundEffects.push_back(sky);

    vkDestroyShaderModule(_device, gradientShader, nullptr);
    vkDestroyShaderModule(_device, skyShader, nullptr);
    _mainDeletionQueue.push_function([=]()
    {
        vkDestroyPipelineLayout(_device, _gradientPipelineLayout, nullptr);
        vkDestroyPipeline(_device, sky.pipeline, nullptr);
        vkDestroyPipeline(_device, gradient.pipeline, nullptr);
    });
}

void VulkanEngine::immediate_submit(std::function<void(VkCommandBuffer cmd)>&& function)
{
    VK_CHECK(vkResetFences(_device, 1, &_immFence));
    VK_CHECK(vkResetCommandBuffer(_immCommandBuffer, 0));

    VkCommandBuffer cmd = _immCommandBuffer;
    VkCommandBufferBeginInfo cmdBeginInfo =
        vkinit::command_buffer_begin_info(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);

    VK_CHECK(vkBeginCommandBuffer(cmd, &cmdBeginInfo));

    function(cmd);

    VK_CHECK(vkEndCommandBuffer(cmd));

    VkCommandBufferSubmitInfo cmdInfo = vkinit::command_buffer_submit_info(cmd);
    VkSubmitInfo2 submit = vkinit::submit_info(&cmdInfo, nullptr, nullptr);

    // Submit command buffer to the queue and execute it
    // _renderFence will now block until the graphic commands finish execution
    VK_CHECK(vkQueueSubmit2(_graphicsQueue, 1, &submit, _immFence));

    VK_CHECK(vkWaitForFences(_device, 1, &_immFence, true, UINT64_MAX));
}

void VulkanEngine::init_imgui()
{
    VkDescriptorPoolSize poolSizes[] = {
        { VK_DESCRIPTOR_TYPE_SAMPLER, 1000 },
        { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1000 },
        { VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1000 },
        { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1000 },
        { VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER, 1000 },
        { VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER, 1000 },
        { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1000 },
        { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1000 },
        { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 1000 },
        { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC, 1000 },
        { VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT, 1000 }
    };

    VkDescriptorPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    poolInfo.maxSets = 1000;
    poolInfo.poolSizeCount = std::size(poolSizes);
    poolInfo.pPoolSizes = poolSizes;

    VkDescriptorPool imguiPool;
    VK_CHECK(vkCreateDescriptorPool(_device, &poolInfo, nullptr, &imguiPool));

    ImGui::CreateContext();
    ImGui_ImplSDL2_InitForVulkan(_window);

    ImGui_ImplVulkan_InitInfo initInfo{};
    initInfo.Instance = _instance;
    initInfo.PhysicalDevice = _chosenGPU;
    initInfo.Device = _device;
    initInfo.Queue = _graphicsQueue;
    initInfo.DescriptorPool = imguiPool;
    initInfo.MinImageCount = 3;
    initInfo.ImageCount = 3;
    initInfo.UseDynamicRendering = true;

    // Dynamic rendering parameters for imgui
    VkPipelineRenderingCreateInfoKHR pipelineRenderingCreateInfo{};
    pipelineRenderingCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
    pipelineRenderingCreateInfo.colorAttachmentCount = 1;
    pipelineRenderingCreateInfo.pColorAttachmentFormats = &_swapchainImageFormat;

    initInfo.PipelineRenderingCreateInfo = pipelineRenderingCreateInfo;
    initInfo.MSAASamples = VK_SAMPLE_COUNT_1_BIT;

    ImGui_ImplVulkan_Init(&initInfo);
    ImGui_ImplVulkan_CreateFontsTexture();

    _mainDeletionQueue.push_function([=]()
    {
        ImGui_ImplVulkan_Shutdown();
        vkDestroyDescriptorPool(_device, imguiPool, nullptr);
    });
}

void VulkanEngine::draw_imgui(VkCommandBuffer cmd, VkImageView targetImageView)
{
    VkRenderingAttachmentInfo colorAttachment = vkinit::attachment_info(targetImageView,
                                                                        nullptr,
                                                                        VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
    VkRenderingInfo renderingInfo = vkinit::rendering_info(_swapchainExtent, &colorAttachment, nullptr);

    vkCmdBeginRendering(cmd, &renderingInfo);

    ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), cmd);

    vkCmdEndRendering(cmd);
}

void VulkanEngine::init_triangle_pipeline()
{
    VkShaderModule triangleFragShader{};
    if (!vkutil::load_shader_module(FRAG_SHADER_TRIANGLE, _device, triangleFragShader))
    {
        fmt::print("Triangle fragment shader loading failed!");
    }

    VkShaderModule triangleVertShader{};
    if (!vkutil::load_shader_module(VERT_SHADER_TRIANGLE, _device, triangleVertShader))
    {
        fmt::print("Triangle fragment shader loading failed!");
    }

    VkPipelineLayoutCreateInfo pipelineLayoutInfo = vkinit::pipeline_layout_create_info();
    VK_CHECK(vkCreatePipelineLayout(_device, &pipelineLayoutInfo, nullptr, &_trianglePipelineLayout));

    PipelineBuilder pipelineBuilder;
    // Use the triangle layout we created
    pipelineBuilder._pipelineLayout = _trianglePipelineLayout;

    pipelineBuilder.set_shaders(triangleVertShader, triangleFragShader);

    // Set to draw triangles
    pipelineBuilder.set_input_topology(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);
    // Set to fill triangles
    pipelineBuilder.set_polygon_mode(VK_POLYGON_MODE_FILL);
    // No backface culling
    pipelineBuilder.set_cull_mode(VK_CULL_MODE_NONE, VK_FRONT_FACE_CLOCKWISE);
    // No multisampling
    pipelineBuilder.set_multisampling_none();
    // No blending
    pipelineBuilder.disable_blending();
    // No depth testing
    pipelineBuilder.disable_depthtest();

    // Set the image format we will draw into, from draw image
    pipelineBuilder.set_color_attachment_format(_drawImage.imageFormat);
    pipelineBuilder.set_depth_format(VK_FORMAT_UNDEFINED);

    _trianglePipeline = pipelineBuilder.build_pipeline(_device);

    vkDestroyShaderModule(_device, triangleFragShader, nullptr);
    vkDestroyShaderModule(_device, triangleVertShader, nullptr);

    _mainDeletionQueue.push_function([&]()
    {
        vkDestroyPipelineLayout(_device, _trianglePipelineLayout, nullptr);
        vkDestroyPipeline(_device, _trianglePipeline, nullptr);
    });
}
