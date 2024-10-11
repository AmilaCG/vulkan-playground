#version 450

#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_buffer_reference : require

#include "input_structures.glsl"

layout (location = 0) out vec3 outNormal;
layout (location = 1) out vec3 outColor;
layout (location = 2) out vec2 outUV;

struct Vertex
{
    vec3 position;
    float uv_x;
    vec3 normal;
    float uv_y;
    vec4 color;
};

layout (buffer_reference, std430) readonly buffer VertexBuffer
{
    Vertex verticies[];
};

// Push constant block
layout (push_constant) uniform constants
{
    mat4 renderMatrix;
    VertexBuffer vertexBuffer;
} PushConstants;

void main()
{
    Vertex vertex = PushConstants.vertexBuffer.verticies[gl_VertexIndex];

    vec4 position = vec4(vertex.position, 1.0f);

    gl_Position = sceneData.viewProj * PushConstants.renderMatrix * position;

    outNormal = (PushConstants.renderMatrix * vec4(vertex.normal, 0.0)).xyz;
    outColor = vertex.color.xyz * materialData.colorFactors.xyz;
    outUV.x = vertex.uv_x;
    outUV.y = vertex.uv_y;
}
