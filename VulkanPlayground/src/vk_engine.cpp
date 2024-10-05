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
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/ext/matrix_clip_space.hpp>
#include <glm/ext/matrix_transform.hpp>

#include "vk_initializers.h"
#include "vk_types.h"
#include "vk_images.h"
#include "vk_pipelines.h"

auto COMP_SHADER_PATH_GRADIENT = "gradient_color.comp.spv";
auto COMP_SHADER_PATH_SKY = "sky.comp.spv";
auto FRAG_SHADER_TRIANGLE = "colored_triangle.frag.spv";
auto VERT_SHADER_TRIANGLE = "colored_triangle.vert.spv";
auto VERT_SHADER_TRIANGLE_MESH = "colored_triangle_mesh.vert.spv";

auto MESH_BASIC = "Assets/basicmesh.glb";

VulkanEngine* loadedEngine = nullptr;
constexpr bool bUseValidationLayers = true;
bool resizeRequested = false;

VulkanEngine& VulkanEngine::Get() { return *loadedEngine; }

void VulkanEngine::init()
{
    // only one engine initialization is allowed with the application.
    assert(loadedEngine == nullptr);
    loadedEngine = this;

    // We initialize SDL and create a window with it.
    SDL_Init(SDL_INIT_VIDEO);

    constexpr auto windowFlags = static_cast<SDL_WindowFlags>(SDL_WINDOW_VULKAN | SDL_WINDOW_RESIZABLE);

    _window = SDL_CreateWindow(
        "Vulkan Engine",
        SDL_WINDOWPOS_UNDEFINED,
        SDL_WINDOWPOS_UNDEFINED,
        _windowExtent.width,
        _windowExtent.height,
        windowFlags);
    fmt::print("{}\n", SDL_GetError());

    init_vulkan();
    init_swapchain();
    init_commands();
    init_sync_structures();
    init_descriptors();
    init_pipelines();
    init_imgui();
    init_default_data();

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
    FrameData& currentFrame = get_current_frame();

    // Wait until the GPU has finished rendering the last frame
    VK_CHECK(vkWaitForFences(_device, 1, &currentFrame._renderFence, true, ONE_SEC_NS));

    currentFrame._deletionQueue.flush();
    currentFrame._frameDescriptors.clear_pools(_device);

    uint32_t swapchainImageIndex;
    // Request image from the swapchain
    VkResult e = vkAcquireNextImageKHR(_device,
                                       _swapchain,
                                       ONE_SEC_NS,
                                       currentFrame._swapchainSemaphore,
                                       nullptr,
                                       &swapchainImageIndex);
    if (e == VK_ERROR_OUT_OF_DATE_KHR)
    {
        resizeRequested = true;
        return;
    }

    VK_CHECK(vkResetFences(_device, 1, &currentFrame._renderFence));

    VkCommandBuffer cmd = currentFrame._mainCommandBuffer;

    // Now that we are sure that the commands finished executing, we can safely reset
    // the command buffer to begin recording again
    VK_CHECK(vkResetCommandBuffer(cmd, 0));

    // Begin the command buffer recording. We will be using this command buffer exactly once.
    VkCommandBufferBeginInfo cmdBeginInfo =
        vkinit::command_buffer_begin_info(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT /* Optional flag */);

    VK_CHECK(vkBeginCommandBuffer(cmd, &cmdBeginInfo));

    _drawExtent.width = std::min(_drawImage.imageExtent.width, _drawImage.imageExtent.width) * _renderScale;
    _drawExtent.height = std::min(_drawImage.imageExtent.height, _drawImage.imageExtent.height) * _renderScale;

    // Make the draw image into general layout so we can write on it. As we overwrite it, we don't
    // care about its old layout
    vkutil::transition_image(cmd,
                             _drawImage.image,
                             VK_IMAGE_LAYOUT_UNDEFINED,
                             VK_IMAGE_LAYOUT_GENERAL);

    draw_background(cmd);

    // Transition draw image into color attachment optimal to draw geometry
    vkutil::transition_image(cmd,
                             _drawImage.image,
                             VK_IMAGE_LAYOUT_GENERAL,
                             VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);

    draw_geometry(cmd);

    // Transition draw image into transfer source layout since we are going to copy it to the swapchain
    vkutil::transition_image(cmd,
                             _drawImage.image,
                             VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
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

    e = vkQueuePresentKHR(_graphicsQueue, &presentInfo);
    if (e == VK_ERROR_OUT_OF_DATE_KHR)
    {
        resizeRequested = true;
    }

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

        if (resizeRequested)
        {
            resize_swapchain();
        }

        // Imgui new frame
        ImGui_ImplVulkan_NewFrame();
        ImGui_ImplSDL2_NewFrame();
        ImGui::NewFrame();

        if (ImGui::Begin("background"))
        {
            ImGui::SliderFloat("Render Scale", &_renderScale, 0.3f, 1.0f);

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

    _depthImage.imageFormat = VK_FORMAT_D32_SFLOAT;
    _depthImage.imageExtent = drawImageExtent;

    VkImageUsageFlags depthImageUsages{};
    depthImageUsages |= VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;

    VkImageCreateInfo dImgInfo = vkinit::image_create_info(_depthImage.imageFormat, depthImageUsages, drawImageExtent);
    vmaCreateImage(_allocator, &dImgInfo, &rImgAllocInfo, &_depthImage.image, &_depthImage.allocation, nullptr);

    VkImageViewCreateInfo dViewInfo = vkinit::imageview_create_info(_depthImage.imageFormat,
                                                                    _depthImage.image,
                                                                    VK_IMAGE_ASPECT_DEPTH_BIT);

    VK_CHECK(vkCreateImageView(_device, &dViewInfo, nullptr, &_depthImage.imageView));

    _mainDeletionQueue.push_function([=]()
    {
        vkDestroyImageView(_device, _drawImage.imageView, nullptr);
        vmaDestroyImage(_allocator, _drawImage.image, _drawImage.allocation);

        vkDestroyImageView(_device, _depthImage.imageView, nullptr);
        vmaDestroyImage(_allocator, _depthImage.image, _depthImage.allocation);
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

void VulkanEngine::draw_geometry(VkCommandBuffer cmd)
{
    // Begin a render pass connected to the draw image
    VkRenderingAttachmentInfo colorAttachment = vkinit::attachment_info(_drawImage.imageView,
                                                                        nullptr,
                                                                        VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);

    VkRenderingAttachmentInfo depthAttachment = vkinit::depth_attachment_info(_depthImage.imageView,
                                                                        VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL);

    VkRenderingInfo renderInfo = vkinit::rendering_info(_drawExtent, &colorAttachment, &depthAttachment);
    vkCmdBeginRendering(cmd, &renderInfo);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, _meshPipeline);

    // Set dynamic viewport and scissor. This should be done after binding to a pipeline.
    VkViewport viewport{};
    viewport.x = 0;
    viewport.y = 0;
    viewport.width = _drawExtent.width;
    viewport.height = _drawExtent.height;
    viewport.minDepth = 0.f;
    viewport.maxDepth = 1.f;
    vkCmdSetViewport(cmd, 0, 1, &viewport);

    VkRect2D scissor{};
    scissor.offset.x = 0;
    scissor.offset.y = 0;
    scissor.extent.width = _drawExtent.width;
    scissor.extent.height = _drawExtent.height;
    vkCmdSetScissor(cmd, 0, 1, &scissor);

    const std::shared_ptr<MeshAsset> selectedMesh = _testMeshes[2];

    GPUDrawPushConstants pushConstants{};
    pushConstants.vertexBuffer = selectedMesh->meshBuffers.vertexBufferAddress;

    glm::mat4 view{1.0f};
    view = glm::translate(view, glm::vec3{0, 0, -5});
    // TODO: Following rotation is not used in the tutorial, but the model is rotated without it. Find out why.
    view = glm::rotate(view, glm::radians(180.0f), glm::vec3{0, 1, 0});
    glm::mat4 projection = glm::perspective(glm::radians(70.0f),
                                            (float)_drawExtent.width / (float)_drawExtent.height,
                                            // TODO: Swap near and far depth values as in the tutorial
                                            0.1f,
                                            10000.0f);
    // Invert Y axis on projection matrix so that to align with OpenGL coordinates as gltf uses OpenGL coordinates
    projection[1][1] *= -1;
    pushConstants.worldMatrix = projection * view;

    vkCmdPushConstants(cmd,
                       _meshPipelineLayout,
                       VK_SHADER_STAGE_VERTEX_BIT,
                       0,
                       sizeof(GPUDrawPushConstants),
                       &pushConstants);

    vkCmdBindIndexBuffer(cmd, selectedMesh->meshBuffers.indexBuffer.buffer, 0, VK_INDEX_TYPE_UINT32);

    vkCmdDrawIndexed(cmd,
                     selectedMesh->surfaces[0].count,
                     1,
                     selectedMesh->surfaces[0].startIndex,
                     0,
                     0);

    // Allocate a new uniform buffer for the scene data
    AllocatedBuffer gpuSceneDataBuffer = create_buffer(sizeof(GPUSceneData),
                                                       VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                                                       VMA_MEMORY_USAGE_CPU_TO_GPU);

    get_current_frame()._deletionQueue.push_function([&, gpuSceneDataBuffer]()
    {
        destroy_buffer(gpuSceneDataBuffer);
    });

    // Write the buffer
    GPUSceneData* sceneUniformData = (GPUSceneData*)gpuSceneDataBuffer.allocation->GetMappedData();
    *sceneUniformData = _sceneData;

    // Create a descriptor set that binds that buffer and update it
    VkDescriptorSet globalDescriptor = get_current_frame()._frameDescriptors.allocate(_device,
        _gpuSceneDataDescriptorLayout);

    DescriptorWriter writer{};
    writer.write_buffer(0,
                        gpuSceneDataBuffer.buffer,
                        sizeof(GPUSceneData),
                        0,
                        VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);

    writer.update_set(_device, globalDescriptor);

    vkCmdEndRendering(cmd);
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

    DescriptorWriter writer{};
    writer.write_image(0,
                       _drawImage.imageView,
                       VK_NULL_HANDLE,
                       VK_IMAGE_LAYOUT_GENERAL,
                       VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);

    writer.update_set(_device, _drawImageDescriptors);

    builder.clear();
    builder.add_binding(0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);
    _gpuSceneDataDescriptorLayout = builder.build(_device, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT);

    _mainDeletionQueue.push_function([&]()
    {
        _globalDescriptorAllocator.destroy_pool(_device);
        vkDestroyDescriptorSetLayout(_device, _drawImageDescriptorLayout, nullptr);
        vkDestroyDescriptorSetLayout(_device, _gpuSceneDataDescriptorLayout, nullptr);
    });

    for (int i = 0; i < FRAME_OVERLAP; i++)
    {
        // Create a descriptor pool
        std::vector<DescriptorAllocatorGrowable::PoolSizeRatio> frameSizes = {
            {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 3},
            {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3},
            {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 3},
            {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 4},
        };

        _frames[i]._frameDescriptors = DescriptorAllocatorGrowable{};
        _frames[i]._frameDescriptors.init(_device, 1000, frameSizes);

        _mainDeletionQueue.push_function([&, i]()
        {
            _frames[i]._frameDescriptors.destroy_pools(_device);
        });
    }
}

void VulkanEngine::init_pipelines()
{
    // Compute pipelines
    init_background_pipelines();

    // Graphics pipelines
    init_mesh_pipeline();
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

AllocatedBuffer VulkanEngine::create_buffer(size_t allocSize, VkBufferUsageFlags usage, VmaMemoryUsage memoryUsage)
{
    VkBufferCreateInfo bufferInfo{};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = allocSize;

    bufferInfo.usage = usage;

    VmaAllocationCreateInfo vmaAllocInfo{};
    vmaAllocInfo.usage = memoryUsage;
    vmaAllocInfo.flags = VMA_ALLOCATION_CREATE_MAPPED_BIT;
    AllocatedBuffer newBuffer{};

    VK_CHECK(vmaCreateBuffer(_allocator,
                             &bufferInfo,
                             &vmaAllocInfo,
                             &newBuffer.buffer,
                             &newBuffer.allocation,
                             &newBuffer.info));

    return newBuffer;
}

void VulkanEngine::destroy_buffer(const AllocatedBuffer& buffer)
{
    vmaDestroyBuffer(_allocator, buffer.buffer, buffer.allocation);
}

GPUMeshBuffers VulkanEngine::upload_mesh(std::span<uint32_t> indices, std::span<Vertex> vertices)
{
    const size_t vertexBufferSize = vertices.size() * sizeof(Vertex);
    const size_t indexBufferSize = indices.size() * sizeof(uint32_t);

    GPUMeshBuffers newSurface{};

    // Allocate vertex buffer in the GPU as a SSBO
    newSurface.vertexBuffer = create_buffer(vertexBufferSize,
                                            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                            VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                                            VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                                            VMA_MEMORY_USAGE_GPU_ONLY);

    // Find the adress of the vertex buffer
    VkBufferDeviceAddressInfo deviceAdressInfo{};
    deviceAdressInfo.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
    deviceAdressInfo.buffer = newSurface.vertexBuffer.buffer;

    newSurface.vertexBufferAddress = vkGetBufferDeviceAddress(_device, &deviceAdressInfo);

    // Allocate index buffer in the GPU
    newSurface.indexBuffer = create_buffer(indexBufferSize,
                                           VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                           VMA_MEMORY_USAGE_GPU_ONLY);

    // As allocated buffers are GPU_ONLY, we will be using a staging buffer to write date into them.
    // This is a very common pattern with Vulkan. GPU_ONLY memory can't be written on CPU, we first write
    // the memory on a temporal staging buffer that is CPU writable, and then execute a copy command to
    // copy this buffer into the GPU buffers.
    AllocatedBuffer stagingBuffer = create_buffer(vertexBufferSize + indexBufferSize,
                                                  VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                                  VMA_MEMORY_USAGE_CPU_ONLY);

    void* data = stagingBuffer.allocation->GetMappedData();

    // Copy vertex buffer
    memcpy(data, vertices.data(), vertexBufferSize);
    // Copy index buffer
    // TODO: Do we really need the char* cast?
    memcpy(static_cast<char*>(data) + vertexBufferSize, indices.data(), indexBufferSize);

    immediate_submit([&](VkCommandBuffer cmd)
    {
        VkBufferCopy vertexCopy{};
        vertexCopy.size = vertexBufferSize;

        vkCmdCopyBuffer(cmd, stagingBuffer.buffer, newSurface.vertexBuffer.buffer, 1, &vertexCopy);

        VkBufferCopy indexCopy{};
        indexCopy.srcOffset = vertexBufferSize;
        indexCopy.size = indexBufferSize;

        vkCmdCopyBuffer(cmd, stagingBuffer.buffer, newSurface.indexBuffer.buffer, 1, &indexCopy);
    });

    destroy_buffer(stagingBuffer);

    return newSurface;
}

void VulkanEngine::init_mesh_pipeline()
{
    VkShaderModule triangleFragShader{};
    if (!vkutil::load_shader_module(FRAG_SHADER_TRIANGLE, _device, triangleFragShader))
    {
        fmt::print("Triangle fragment shader loading failed!");
    }

    VkShaderModule triangleVertShader{};
    if (!vkutil::load_shader_module(VERT_SHADER_TRIANGLE_MESH, _device, triangleVertShader))
    {
        fmt::print("Triangle vertex shader loading failed!");
    }

    VkPushConstantRange bufferRange{};
    bufferRange.size = sizeof(GPUDrawPushConstants);
    bufferRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;

    VkPipelineLayoutCreateInfo pipelineLayoutInfo = vkinit::pipeline_layout_create_info();
    pipelineLayoutInfo.pPushConstantRanges = &bufferRange;
    pipelineLayoutInfo.pushConstantRangeCount = 1;

    VK_CHECK(vkCreatePipelineLayout(_device, &pipelineLayoutInfo, nullptr, &_meshPipelineLayout));

    PipelineBuilder pipelineBuilder;

    pipelineBuilder._pipelineLayout = _meshPipelineLayout;
    pipelineBuilder.set_shaders(triangleVertShader, triangleFragShader);
    pipelineBuilder.set_input_topology(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);
    pipelineBuilder.set_polygon_mode(VK_POLYGON_MODE_FILL);
    pipelineBuilder.set_cull_mode(VK_CULL_MODE_NONE, VK_FRONT_FACE_CLOCKWISE);
    pipelineBuilder.set_multisampling_none();
    pipelineBuilder.enable_blending_additive();
    pipelineBuilder.enable_depthtest(true, VK_COMPARE_OP_GREATER_OR_EQUAL);
    pipelineBuilder.set_color_attachment_format(_drawImage.imageFormat);
    pipelineBuilder.set_depth_format(_depthImage.imageFormat);

    _meshPipeline = pipelineBuilder.build_pipeline(_device);

    vkDestroyShaderModule(_device, triangleFragShader, nullptr);
    vkDestroyShaderModule(_device, triangleVertShader, nullptr);

    _mainDeletionQueue.push_function([&]()
    {
        vkDestroyPipelineLayout(_device, _meshPipelineLayout, nullptr);
        vkDestroyPipeline(_device, _meshPipeline, nullptr);
    });
}

void VulkanEngine::init_default_data()
{
    _testMeshes = load_gltf_meshes(this, MESH_BASIC).value();

    _mainDeletionQueue.push_function([&]()
    {
        for (const std::shared_ptr<MeshAsset>& mesh : _testMeshes)
        {
            destroy_buffer(mesh->meshBuffers.indexBuffer);
            destroy_buffer(mesh->meshBuffers.vertexBuffer);
        }
    });
}

void VulkanEngine::resize_swapchain()
{
    vkDeviceWaitIdle(_device);

    destroy_swapchain();

    int width, height;
    SDL_GetWindowSize(_window, &width, &height);
    _windowExtent.width = width;
    _windowExtent.height = height;

    create_swapchain(_windowExtent.width, _windowExtent.height);

    resizeRequested = false;
}
