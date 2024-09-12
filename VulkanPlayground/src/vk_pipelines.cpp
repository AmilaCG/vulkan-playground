#include "vk_pipelines.h"

#include <fstream>
#include "vk_initializers.h"

bool vkutil::load_shader_module(const char* filePath, VkDevice device, VkShaderModule& outShaderModule)
{
    // Open file with cursor at the end
    std::ifstream file(filePath, std::ios::ate | std::ios::binary);

    if (!file.is_open())
    {
        return false;
    }

    const size_t fileSize = file.tellg();
    // SPIRV expects the buffer to be on uint32
    std::vector<uint32_t> buffer(fileSize / sizeof(uint32_t));

    // Put the cursor at the begining
    file.seekg(0);
    file.read(reinterpret_cast<char*>(buffer.data()), fileSize);
    file.close();

    VkShaderModuleCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    // codeSize has to be in bytes
    createInfo.codeSize = buffer.size() * sizeof(uint32_t);
    createInfo.pCode = buffer.data();

    VkShaderModule shaderModule;
    if (vkCreateShaderModule(device, &createInfo, nullptr, &shaderModule) != VK_SUCCESS)
    {
        return false;
    }

    outShaderModule = shaderModule;
    return true;
}
