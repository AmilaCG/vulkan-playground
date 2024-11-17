#include "vk_loader.h"

#include <iostream>
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/quaternion.hpp>
#include <fastgltf/core.hpp>
#include <fastgltf/tools.hpp>
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#include "vk_engine.h"
#include "vk_types.h"

static VkFilter extract_filter(const fastgltf::Filter filter)
{
    switch (filter)
    {
        // Nearest samplers
        case fastgltf::Filter::Nearest:
        case fastgltf::Filter::NearestMipMapNearest:
        case fastgltf::Filter::NearestMipMapLinear:
            return VK_FILTER_NEAREST;

        // Linear samplers
        case fastgltf::Filter::Linear:
        case fastgltf::Filter::LinearMipMapNearest:
        case fastgltf::Filter::LinearMipMapLinear:
        default:
            return VK_FILTER_LINEAR;
    }
}

static VkSamplerMipmapMode extract_mipmap_mode(const fastgltf::Filter filter)
{
    switch (filter)
    {
        case fastgltf::Filter::NearestMipMapNearest:
        case fastgltf::Filter::LinearMipMapNearest:
            return VK_SAMPLER_MIPMAP_MODE_NEAREST;

        case fastgltf::Filter::NearestMipMapLinear:
        case fastgltf::Filter::LinearMipMapLinear:
        default:
            return VK_SAMPLER_MIPMAP_MODE_LINEAR;
    }
}

void LoadedGLTF::draw(const glm::mat4& topMatrix, DrawContext& ctx)
{
    for (auto& node : topNodes)
    {
        node->draw(topMatrix, ctx);
    }
}

void LoadedGLTF::clearAll()
{
    VkDevice device = creator->_device;

    descriptorPool.destroy_pools(device);
    creator->destroy_buffer(materialDataBuffer);

    for (auto& [k, v]: meshes)
    {
        creator->destroy_buffer(v->meshBuffers.indexBuffer);
        creator->destroy_buffer(v->meshBuffers.vertexBuffer);
    }

    for (auto& [k, v]: images)
    {
        if (v.image == creator->_errorCheckboardImage.image)
        {
            //dont destroy the default images
            continue;
        }
        creator->destroy_image(v);
    }

    for (auto& sampler: samplers)
    {
        vkDestroySampler(device, sampler, nullptr);
    }
}

std::optional<AllocatedImage> load_image(VulkanEngine* engine, fastgltf::Asset& asset, fastgltf::Image& image)
{
    AllocatedImage newImage{};

    int width, height, nrChannels;

    std::visit(
        fastgltf::visitor{
            [](auto& arg) {},
            [&](fastgltf::sources::URI& filePath)
            {
                assert(filePath.fileByteOffset == 0); // We don't support offsets with stbi.
                assert(filePath.uri.isLocalPath()); // We're only capable of loading local files.

                const std::string path(filePath.uri.path().begin(),
                                       filePath.uri.path().end()); // Thanks C++.
                unsigned char* data = stbi_load(path.c_str(), &width, &height, &nrChannels, 4);
                if (data)
                {
                    VkExtent3D imagesize;
                    imagesize.width = width;
                    imagesize.height = height;
                    imagesize.depth = 1;

                    newImage = engine->create_image(data, imagesize, VK_FORMAT_R8G8B8A8_UNORM,
                                                    VK_IMAGE_USAGE_SAMPLED_BIT, true);

                    stbi_image_free(data);
                }
            },
            [&](fastgltf::sources::Vector& vector)
            {
                unsigned char* data = stbi_load_from_memory(vector.bytes.data(),
                                                            static_cast<int>(vector.bytes.size()),
                                                            &width,
                                                            &height,
                                                            &nrChannels,
                                                            4);
                if (data)
                {
                    VkExtent3D imagesize;
                    imagesize.width = width;
                    imagesize.height = height;
                    imagesize.depth = 1;

                    newImage = engine->create_image(data, imagesize, VK_FORMAT_R8G8B8A8_UNORM,
                                                    VK_IMAGE_USAGE_SAMPLED_BIT, true);

                    stbi_image_free(data);
                }
            },
            [&](fastgltf::sources::BufferView& view)
            {
                auto& bufferView = asset.bufferViews[view.bufferViewIndex];
                auto& buffer = asset.buffers[bufferView.bufferIndex];

                std::visit(fastgltf::visitor{
                               // We only care about VectorWithMime here, because we specify LoadExternalBuffers,
                               // meaning all buffers are already loaded into a vector.
                               [](auto& arg) {},
                               [&](fastgltf::sources::Array& array)
                               {
                                   unsigned char* data = stbi_load_from_memory(
                                       array.bytes.data() + bufferView.byteOffset,
                                       static_cast<int>(bufferView.byteLength),
                                       &width, &height, &nrChannels, 4);
                                   if (data)
                                   {
                                       VkExtent3D imagesize;
                                       imagesize.width = width;
                                       imagesize.height = height;
                                       imagesize.depth = 1;

                                       newImage = engine->create_image(data, imagesize, VK_FORMAT_R8G8B8A8_UNORM,
                                                                       VK_IMAGE_USAGE_SAMPLED_BIT, true);

                                       stbi_image_free(data);
                                   }
                               }
                           },
                           buffer.data);
            },
        },
        image.data);

    // if any of the attempts to load the data failed, we havent written the image
    // so handle is null
    if (newImage.image == VK_NULL_HANDLE)
    {
        return {};
    }
    else
    {
        return newImage;
    }
}

std::optional<std::shared_ptr<LoadedGLTF>> load_gltf(VulkanEngine* engine, std::string_view filePath)
{
    fmt::print("Loading GLTF: {}\n", filePath);

    std::shared_ptr<LoadedGLTF> scene = std::make_shared<LoadedGLTF>();
    scene->creator = engine;
    LoadedGLTF& file = *scene.get();

    fastgltf::Parser parser{};

    constexpr auto gltfOptions = fastgltf::Options::DontRequireValidAssetMember |
                                 fastgltf::Options::AllowDouble |
                                 fastgltf::Options::LoadGLBBuffers |
                                 fastgltf::Options::LoadExternalBuffers;

    fastgltf::GltfDataBuffer data;
    data.loadFromFile(filePath);

    fastgltf::Asset gltf;

    std::filesystem::path path = filePath;

    auto type = fastgltf::determineGltfFileType(&data);
    if (type == fastgltf::GltfType::glTF)
    {
        auto load = parser.loadGltf(&data, path.parent_path(), gltfOptions);
        if (load)
        {
            gltf = std::move(load.get());
        }
        else
        {
            std::cerr << "Failed to load glTF: " << fastgltf::to_underlying(load.error()) << std::endl;
            return {};
        }
    }
    else if (type == fastgltf::GltfType::GLB)
    {
        auto load = parser.loadGltfBinary(&data, path.parent_path(), gltfOptions);
        if (load)
        {
            gltf = std::move(load.get());
        }
        else
        {
            std::cerr << "Failed to load glTF: " << fastgltf::to_underlying(load.error()) << std::endl;
            return {};
        }
    }
    else
    {
        std::cerr << "Failed to determine glTF container" << std::endl;
        return {};
    }

    // We can estimate the descriptors we will need accurately
    std::vector<DescriptorAllocatorGrowable::PoolSizeRatio> sizes = {
        { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 3 },
        { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 3 },
        { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1 }
    };
    file.descriptorPool.init(engine->_device, gltf.materials.size(), sizes);

    // Load samplers
    for (fastgltf::Sampler& sampler : gltf.samplers)
    {
        VkSamplerCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        createInfo.pNext = nullptr;
        createInfo.maxLod = VK_LOD_CLAMP_NONE;
        createInfo.minLod = 0;

        createInfo.magFilter = extract_filter(sampler.magFilter.value_or(fastgltf::Filter::Nearest));
        createInfo.minFilter = extract_filter(sampler.minFilter.value_or(fastgltf::Filter::Nearest));

        createInfo.mipmapMode = extract_mipmap_mode(sampler.minFilter.value_or(fastgltf::Filter::Nearest));

        VkSampler newSampler;
        vkCreateSampler(engine->_device, &createInfo, nullptr, &newSampler);

        file.samplers.push_back(newSampler);
    }

    std::vector<std::shared_ptr<MeshAsset>> meshes;
    std::vector<std::shared_ptr<Node>> nodes;
    std::vector<AllocatedImage> images;
    std::vector<std::shared_ptr<GLTFMaterial>> materials;

    // Load all textures
    for (fastgltf::Image& image : gltf.images)
    {
        std::optional<AllocatedImage> img = load_image(engine, gltf, image);

        if (img.has_value())
        {
            images.push_back(*img);
            file.images[image.name.c_str()] = *img;
        }
        else
        {
            // Loading failed
            images.push_back(engine->_errorCheckboardImage);
            std::cout << "GLTF failed to load texture " << image.name << std::endl;
        }
    }

    // Create buffer to hold the material data
    file.materialDataBuffer = engine->create_buffer(
        sizeof(GLTFMetallicRoughness::MaterialConstants) * gltf.materials.size(),
        VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
        VMA_MEMORY_USAGE_CPU_TO_GPU);
    int dataIndex = 0;
    GLTFMetallicRoughness::MaterialConstants* sceneMaterialConstants =
        static_cast<GLTFMetallicRoughness::MaterialConstants*>(file.materialDataBuffer.info.pMappedData);

    for (fastgltf::Material& mat : gltf.materials)
    {
        std::shared_ptr<GLTFMaterial> newMat = std::make_shared<GLTFMaterial>();
        materials.push_back(newMat);
        file.materials[mat.name.c_str()] = newMat;

        GLTFMetallicRoughness::MaterialConstants constants{};
        constants.colorFactors.x = mat.pbrData.baseColorFactor[0];
        constants.colorFactors.y = mat.pbrData.baseColorFactor[1];
        constants.colorFactors.z = mat.pbrData.baseColorFactor[2];
        constants.colorFactors.w = mat.pbrData.baseColorFactor[3];

        constants.metalRoughFactors.x = mat.pbrData.metallicFactor;
        constants.metalRoughFactors.y = mat.pbrData.roughnessFactor;
        // Write material parameters to buffer
        sceneMaterialConstants[dataIndex] = constants;

        MaterialPass passType = MaterialPass::MainColor;
        if (mat.alphaMode == fastgltf::AlphaMode::Blend)
        {
            passType = MaterialPass::Transparent;
        }

        GLTFMetallicRoughness::MaterialResources materialResources;
        // Default the material textures
        materialResources.colorImage = engine->_whiteImage;
        materialResources.colorSampler = engine->_defaultSamplerLinear;
        materialResources.metalRoughImage = engine->_whiteImage;
        materialResources.metalRoughSampler = engine->_defaultSamplerLinear;

        // Set the uniform buffer for the material data
        materialResources.dataBuffer = file.materialDataBuffer.buffer;
        materialResources.dataBufferOffset = dataIndex * sizeof(GLTFMetallicRoughness::MaterialConstants);
        // Grab textures from gltf file
        if (mat.pbrData.baseColorTexture.has_value()) {
            auto texture = gltf.textures[mat.pbrData.baseColorTexture.value().textureIndex];
            if (texture.imageIndex.has_value())
            {
                size_t img = texture.imageIndex.value();
                materialResources.colorImage = images[img];
            }
            if (texture.samplerIndex.has_value())
            {
                size_t sampler = texture.samplerIndex.value();
                materialResources.colorSampler = file.samplers[sampler];
            }
        }
        // Build material
        newMat->data = engine->_metalRoughMaterial.write_material(engine->_device,
                                                                  passType,
                                                                  materialResources,
                                                                  file.descriptorPool);

        dataIndex++;
    }

    // use the same vectors for all meshes so that the memory doesnt reallocate as often
    std::vector<uint32_t> indices;
    std::vector<Vertex> vertices;

    for (fastgltf::Mesh& mesh: gltf.meshes)
    {
        std::shared_ptr<MeshAsset> newmesh = std::make_shared<MeshAsset>();
        meshes.push_back(newmesh);
        file.meshes[mesh.name.c_str()] = newmesh;
        newmesh->name = mesh.name;

        // clear the mesh arrays each mesh, we dont want to merge them by error
        indices.clear();
        vertices.clear();

        for (auto&& p: mesh.primitives)
        {
            GeoSurface newSurface;
            newSurface.startIndex = (uint32_t) indices.size();
            newSurface.count = (uint32_t) gltf.accessors[p.indicesAccessor.value()].count;

            size_t initialVertex = vertices.size();

            // load indexes
            {
                fastgltf::Accessor& indexaccessor = gltf.accessors[p.indicesAccessor.value()];
                indices.reserve(indices.size() + indexaccessor.count);

                fastgltf::iterateAccessor<std::uint32_t>(gltf, indexaccessor,
                                                         [&](std::uint32_t idx)
                                                         {
                                                             indices.push_back(idx + initialVertex);
                                                         });
            }

            // load vertex positions
            {
                fastgltf::Accessor& posAccessor = gltf.accessors[p.findAttribute("POSITION")->second];
                vertices.resize(vertices.size() + posAccessor.count);

                fastgltf::iterateAccessorWithIndex<glm::vec3>(gltf, posAccessor,
                                                              [&](glm::vec3 v, size_t index)
                                                              {
                                                                  Vertex newvtx;
                                                                  newvtx.position = v;
                                                                  newvtx.normal = {1, 0, 0};
                                                                  newvtx.color = glm::vec4{1.f};
                                                                  newvtx.uv_x = 0;
                                                                  newvtx.uv_y = 0;
                                                                  vertices[initialVertex + index] = newvtx;
                                                              });
            }

            // load vertex normals
            auto normals = p.findAttribute("NORMAL");
            if (normals != p.attributes.end())
            {
                fastgltf::iterateAccessorWithIndex<glm::vec3>(gltf, gltf.accessors[(*normals).second],
                                                              [&](glm::vec3 v, size_t index)
                                                              {
                                                                  vertices[initialVertex + index].normal = v;
                                                              });
            }

            // load UVs
            auto uv = p.findAttribute("TEXCOORD_0");
            if (uv != p.attributes.end())
            {
                fastgltf::iterateAccessorWithIndex<glm::vec2>(gltf, gltf.accessors[(*uv).second],
                                                              [&](glm::vec2 v, size_t index)
                                                              {
                                                                  vertices[initialVertex + index].uv_x = v.x;
                                                                  vertices[initialVertex + index].uv_y = v.y;
                                                              });
            }

            // load vertex colors
            auto colors = p.findAttribute("COLOR_0");
            if (colors != p.attributes.end())
            {
                fastgltf::iterateAccessorWithIndex<glm::vec4>(gltf, gltf.accessors[(*colors).second],
                                                              [&](glm::vec4 v, size_t index)
                                                              {
                                                                  vertices[initialVertex + index].color = v;
                                                              });
            }

            if (p.materialIndex.has_value())
            {
                newSurface.material = materials[p.materialIndex.value()];
            }
            else
            {
                newSurface.material = materials[0];
            }

            glm::vec3 minPos = vertices[initialVertex].position;
            glm::vec3 maxPos = vertices[initialVertex].position;
            // Iterate through vertices of this surface and find min/max bounds
            for (int i = initialVertex; i < vertices.size(); i++)
            {
                minPos = glm::min(minPos, vertices[i].position);
                maxPos = glm::max(maxPos, vertices[i].position);
            }

            newSurface.bounds.origin = (maxPos + minPos) / 2.0f;
            newSurface.bounds.extents = (maxPos - minPos) / 2.0f;
            newSurface.bounds.sphereRadius = glm::length(newSurface.bounds.extents);

            newmesh->surfaces.push_back(newSurface);
        }

        newmesh->meshBuffers = engine->upload_mesh(indices, vertices);
    }

    // load all nodes and their meshes
    for (fastgltf::Node& node : gltf.nodes)
    {
        std::shared_ptr<Node> newNode;

        // find if the node has a mesh, and if it does hook it to the mesh pointer and allocate it with the meshnode class
        if (node.meshIndex.has_value())
        {
            newNode = std::make_shared<MeshNode>();
            static_cast<MeshNode *>(newNode.get())->mesh = meshes[*node.meshIndex];
        }
        else
        {
            newNode = std::make_shared<Node>();
        }

        nodes.push_back(newNode);
        file.nodes[node.name.c_str()];

        std::visit(fastgltf::visitor{
                       [&](fastgltf::Node::TransformMatrix matrix)
                       {
                           memcpy(&newNode->localTransform, matrix.data(), sizeof(matrix));
                       },
                       [&](fastgltf::TRS transform)
                       {
                           glm::vec3 tl(transform.translation[0], transform.translation[1],
                                        transform.translation[2]);
                           glm::quat rot(transform.rotation[3], transform.rotation[0], transform.rotation[1],
                                         transform.rotation[2]);
                           glm::vec3 sc(transform.scale[0], transform.scale[1], transform.scale[2]);

                           glm::mat4 tm = glm::translate(glm::mat4(1.f), tl);
                           glm::mat4 rm = glm::toMat4(rot);
                           glm::mat4 sm = glm::scale(glm::mat4(1.f), sc);

                           newNode->localTransform = tm * rm * sm;
                       }
                   },
                   node.transform);
    }

    // run loop again to setup transform hierarchy
    for (int i = 0; i < gltf.nodes.size(); i++)
    {
        fastgltf::Node& node = gltf.nodes[i];
        std::shared_ptr<Node>& sceneNode = nodes[i];

        for (auto& c: node.children)
        {
            sceneNode->children.push_back(nodes[c]);
            nodes[c]->parent = sceneNode;
        }
    }

    // find the top nodes, with no parents
    for (auto& node: nodes)
    {
        if (node->parent.lock() == nullptr)
        {
            file.topNodes.push_back(node);
            node->refresh_transform(glm::mat4{1.f});
        }
    }

    return scene;
}
