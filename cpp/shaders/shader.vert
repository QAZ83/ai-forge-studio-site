#version 450

// Vertex shader for AI Forge Studio - Vulkan Renderer
// Author: M.3R3

// Input vertex attributes
layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec2 inTexCoord;
layout(location = 3) in vec3 inColor;

// Output to fragment shader
layout(location = 0) out vec3 fragPosition;
layout(location = 1) out vec3 fragNormal;
layout(location = 2) out vec2 fragTexCoord;
layout(location = 3) out vec3 fragColor;

// Uniform buffer - per-frame data
layout(set = 0, binding = 0) uniform UniformBufferObject {
    mat4 model;
    mat4 view;
    mat4 projection;
    mat4 normalMatrix;
    vec3 cameraPosition;
    float time;
    vec4 lightPositions[4];
    vec4 lightColors[4];
    int lightCount;
    float ambientStrength;
    float specularStrength;
    float padding;
} ubo;

// Push constants for per-object data
layout(push_constant) uniform PushConstants {
    mat4 modelMatrix;
    vec4 objectColor;
} push;

void main() {
    // Transform vertex position to world space
    vec4 worldPosition = push.modelMatrix * vec4(inPosition, 1.0);
    fragPosition = worldPosition.xyz;
    
    // Transform normal to world space (using normal matrix for proper scaling)
    mat3 normalMat = mat3(ubo.normalMatrix);
    fragNormal = normalize(normalMat * inNormal);
    
    // Pass through texture coordinates
    fragTexCoord = inTexCoord;
    
    // Vertex color (can be modulated by object color)
    fragColor = inColor * push.objectColor.rgb;
    
    // Final clip-space position
    gl_Position = ubo.projection * ubo.view * worldPosition;
}
