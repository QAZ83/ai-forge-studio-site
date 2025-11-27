#version 450

// Fragment shader for AI Forge Studio - Vulkan Renderer
// Author: M.3R3

// Input from vertex shader
layout(location = 0) in vec3 fragPosition;
layout(location = 1) in vec3 fragNormal;
layout(location = 2) in vec2 fragTexCoord;
layout(location = 3) in vec3 fragColor;

// Output color
layout(location = 0) out vec4 outColor;

// Uniform buffer
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

// Texture sampler
layout(set = 1, binding = 0) uniform sampler2D texSampler;

// Constants
const float PI = 3.14159265359;

// PBR Helper Functions
float DistributionGGX(vec3 N, vec3 H, float roughness) {
    float a = roughness * roughness;
    float a2 = a * a;
    float NdotH = max(dot(N, H), 0.0);
    float NdotH2 = NdotH * NdotH;
    
    float nom = a2;
    float denom = (NdotH2 * (a2 - 1.0) + 1.0);
    denom = PI * denom * denom;
    
    return nom / denom;
}

float GeometrySchlickGGX(float NdotV, float roughness) {
    float r = (roughness + 1.0);
    float k = (r * r) / 8.0;
    
    float nom = NdotV;
    float denom = NdotV * (1.0 - k) + k;
    
    return nom / denom;
}

float GeometrySmith(vec3 N, vec3 V, vec3 L, float roughness) {
    float NdotV = max(dot(N, V), 0.0);
    float NdotL = max(dot(N, L), 0.0);
    float ggx2 = GeometrySchlickGGX(NdotV, roughness);
    float ggx1 = GeometrySchlickGGX(NdotL, roughness);
    
    return ggx1 * ggx2;
}

vec3 fresnelSchlick(float cosTheta, vec3 F0) {
    return F0 + (1.0 - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

// Main fragment shader
void main() {
    // Get base color from texture or vertex color
    vec4 texColor = texture(texSampler, fragTexCoord);
    vec3 albedo = texColor.rgb * fragColor;
    
    // Material properties (can be made uniform later)
    float metallic = 0.0;
    float roughness = 0.5;
    float ao = 1.0;
    
    // Normal
    vec3 N = normalize(fragNormal);
    
    // View direction
    vec3 V = normalize(ubo.cameraPosition - fragPosition);
    
    // Calculate reflectance at normal incidence
    vec3 F0 = vec3(0.04);
    F0 = mix(F0, albedo, metallic);
    
    // Lighting accumulator
    vec3 Lo = vec3(0.0);
    
    // Process each light
    for (int i = 0; i < ubo.lightCount; ++i) {
        vec3 lightPos = ubo.lightPositions[i].xyz;
        vec3 lightColor = ubo.lightColors[i].rgb;
        float lightIntensity = ubo.lightColors[i].a;
        
        // Light direction and distance
        vec3 L = normalize(lightPos - fragPosition);
        vec3 H = normalize(V + L);
        float distance = length(lightPos - fragPosition);
        float attenuation = 1.0 / (distance * distance);
        vec3 radiance = lightColor * lightIntensity * attenuation;
        
        // Cook-Torrance BRDF
        float NDF = DistributionGGX(N, H, roughness);
        float G = GeometrySmith(N, V, L, roughness);
        vec3 F = fresnelSchlick(max(dot(H, V), 0.0), F0);
        
        vec3 numerator = NDF * G * F;
        float denominator = 4.0 * max(dot(N, V), 0.0) * max(dot(N, L), 0.0) + 0.0001;
        vec3 specular = numerator / denominator;
        
        // Energy conservation
        vec3 kS = F;
        vec3 kD = vec3(1.0) - kS;
        kD *= 1.0 - metallic;
        
        // Final radiance
        float NdotL = max(dot(N, L), 0.0);
        Lo += (kD * albedo / PI + specular) * radiance * NdotL;
    }
    
    // Ambient lighting
    vec3 ambient = vec3(ubo.ambientStrength) * albedo * ao;
    
    // Final color
    vec3 color = ambient + Lo;
    
    // HDR tonemapping (Reinhard)
    color = color / (color + vec3(1.0));
    
    // Gamma correction
    color = pow(color, vec3(1.0 / 2.2));
    
    outColor = vec4(color, texColor.a);
}
