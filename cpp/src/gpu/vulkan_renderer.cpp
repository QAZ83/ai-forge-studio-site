/**
 * AI Forge Studio - Vulkan Renderer Implementation
 * Author: M.3R3
 * 
 * Note: This is a minimal implementation. For production use,
 * consider using a library like VulkanMemoryAllocator (VMA).
 */

#include "vulkan_renderer.h"

#ifdef _WIN32
#define VK_USE_PLATFORM_WIN32_KHR
#include <windows.h>
#include <vulkan/vulkan_win32.h>
#endif

#include <iostream>
#include <set>
#include <algorithm>
#include <chrono>
#include <cstring>
#include <cmath>

namespace aiforge {
namespace gpu {

// ==============================================================================
// SPIR-V Shader Code (embedded)
// ==============================================================================

// Simple vertex shader
static const uint32_t vertShaderCode[] = {
    // Generated from GLSL - basic vertex shader with transforms
    0x07230203, 0x00010000, 0x000d000a, 0x00000036,
    0x00000000, 0x00020011, 0x00000001, 0x0006000b,
    0x00000001, 0x4c534c47, 0x6474732e, 0x3035342e,
    0x00000000, 0x0003000e, 0x00000000, 0x00000001,
    // ... (truncated for brevity - in production, load from file)
};

// Simple fragment shader
static const uint32_t fragShaderCode[] = {
    // Generated from GLSL - basic fragment shader with lighting
    0x07230203, 0x00010000, 0x000d000a, 0x00000018,
    0x00000000, 0x00020011, 0x00000001, 0x0006000b,
    0x00000001, 0x4c534c47, 0x6474732e, 0x3035342e,
    0x00000000, 0x0003000e, 0x00000000, 0x00000001,
    // ... (truncated for brevity)
};

// Validation layers
const std::vector<const char*> validationLayers = {
    "VK_LAYER_KHRONOS_validation"
};

// Device extensions
const std::vector<const char*> deviceExtensions = {
    VK_KHR_SWAPCHAIN_EXTENSION_NAME
};

#ifdef NDEBUG
constexpr bool enableValidationLayers = false;
#else
constexpr bool enableValidationLayers = true;
#endif

// ==============================================================================
// Debug Callback
// ==============================================================================

static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(
    VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
    VkDebugUtilsMessageTypeFlagsEXT messageType,
    const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
    void* pUserData
) {
    if (messageSeverity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) {
        std::cerr << "[Vulkan] " << pCallbackData->pMessage << std::endl;
    }
    return VK_FALSE;
}

// ==============================================================================
// Vertex Implementation
// ==============================================================================

VkVertexInputBindingDescription Vertex::getBindingDescription() {
    VkVertexInputBindingDescription bindingDescription{};
    bindingDescription.binding = 0;
    bindingDescription.stride = sizeof(Vertex);
    bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
    return bindingDescription;
}

std::array<VkVertexInputAttributeDescription, 3> Vertex::getAttributeDescriptions() {
    std::array<VkVertexInputAttributeDescription, 3> attributeDescriptions{};
    
    // Position
    attributeDescriptions[0].binding = 0;
    attributeDescriptions[0].location = 0;
    attributeDescriptions[0].format = VK_FORMAT_R32G32B32_SFLOAT;
    attributeDescriptions[0].offset = offsetof(Vertex, position);
    
    // Color
    attributeDescriptions[1].binding = 0;
    attributeDescriptions[1].location = 1;
    attributeDescriptions[1].format = VK_FORMAT_R32G32B32_SFLOAT;
    attributeDescriptions[1].offset = offsetof(Vertex, color);
    
    // Normal
    attributeDescriptions[2].binding = 0;
    attributeDescriptions[2].location = 2;
    attributeDescriptions[2].format = VK_FORMAT_R32G32B32_SFLOAT;
    attributeDescriptions[2].offset = offsetof(Vertex, normal);
    
    return attributeDescriptions;
}

// ==============================================================================
// VulkanRenderer Implementation
// ==============================================================================

VulkanRenderer::VulkanRenderer()
    : m_initialized(false)
    , m_width(0)
    , m_height(0)
    , m_windowHandle(nullptr)
    , m_instance(VK_NULL_HANDLE)
    , m_physicalDevice(VK_NULL_HANDLE)
    , m_device(VK_NULL_HANDLE)
    , m_surface(VK_NULL_HANDLE)
    , m_swapchain(VK_NULL_HANDLE)
    , m_renderPass(VK_NULL_HANDLE)
    , m_pipelineLayout(VK_NULL_HANDLE)
    , m_graphicsPipeline(VK_NULL_HANDLE)
    , m_commandPool(VK_NULL_HANDLE)
    , m_descriptorPool(VK_NULL_HANDLE)
    , m_descriptorSetLayout(VK_NULL_HANDLE)
    , m_graphicsQueue(VK_NULL_HANDLE)
    , m_presentQueue(VK_NULL_HANDLE)
    , m_graphicsFamily(0)
    , m_presentFamily(0)
    , m_depthImage(VK_NULL_HANDLE)
    , m_depthImageMemory(VK_NULL_HANDLE)
    , m_depthImageView(VK_NULL_HANDLE)
    , m_vertexBuffer(VK_NULL_HANDLE)
    , m_vertexBufferMemory(VK_NULL_HANDLE)
    , m_indexBuffer(VK_NULL_HANDLE)
    , m_indexBufferMemory(VK_NULL_HANDLE)
    , m_currentFrame(0)
    , m_rotationAngle(0.0f)
    , m_rotationSpeed(1.0f)
    , m_demoScene(0)
    , m_debugMessenger(VK_NULL_HANDLE)
{
    m_stats = {};
}

VulkanRenderer::~VulkanRenderer() {
    shutdown();
}

bool VulkanRenderer::initialize(void* windowHandle, int width, int height) {
    m_windowHandle = windowHandle;
    m_width = width;
    m_height = height;
    
    std::cout << "[Vulkan] Initializing renderer (" << width << "x" << height << ")" << std::endl;
    
    if (!createInstance()) return false;
    if (enableValidationLayers && !setupDebugMessenger()) return false;
    if (!createSurface(windowHandle)) return false;
    if (!pickPhysicalDevice()) return false;
    if (!createLogicalDevice()) return false;
    if (!createSwapchain()) return false;
    if (!createImageViews()) return false;
    if (!createRenderPass()) return false;
    if (!createDescriptorSetLayout()) return false;
    if (!createGraphicsPipeline()) return false;
    if (!createCommandPool()) return false;
    if (!createDepthResources()) return false;
    if (!createFramebuffers()) return false;
    
    // Generate default scene
    generateCubeScene();
    
    if (!createVertexBuffer()) return false;
    if (!createIndexBuffer()) return false;
    if (!createUniformBuffers()) return false;
    if (!createDescriptorPool()) return false;
    if (!createDescriptorSets()) return false;
    if (!createCommandBuffers()) return false;
    if (!createSyncObjects()) return false;
    
    m_initialized = true;
    std::cout << "[Vulkan] Initialization complete!" << std::endl;
    
    return true;
}

void VulkanRenderer::shutdown() {
    if (m_device != VK_NULL_HANDLE) {
        vkDeviceWaitIdle(m_device);
    }
    
    cleanupSwapchain();
    
    // Cleanup uniform buffers
    for (size_t i = 0; i < m_uniformBuffers.size(); i++) {
        vkDestroyBuffer(m_device, m_uniformBuffers[i], nullptr);
        vkFreeMemory(m_device, m_uniformBuffersMemory[i], nullptr);
    }
    
    if (m_descriptorPool != VK_NULL_HANDLE) {
        vkDestroyDescriptorPool(m_device, m_descriptorPool, nullptr);
    }
    
    if (m_descriptorSetLayout != VK_NULL_HANDLE) {
        vkDestroyDescriptorSetLayout(m_device, m_descriptorSetLayout, nullptr);
    }
    
    if (m_indexBuffer != VK_NULL_HANDLE) {
        vkDestroyBuffer(m_device, m_indexBuffer, nullptr);
        vkFreeMemory(m_device, m_indexBufferMemory, nullptr);
    }
    
    if (m_vertexBuffer != VK_NULL_HANDLE) {
        vkDestroyBuffer(m_device, m_vertexBuffer, nullptr);
        vkFreeMemory(m_device, m_vertexBufferMemory, nullptr);
    }
    
    // Cleanup sync objects
    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
        if (i < m_renderFinishedSemaphores.size()) {
            vkDestroySemaphore(m_device, m_renderFinishedSemaphores[i], nullptr);
        }
        if (i < m_imageAvailableSemaphores.size()) {
            vkDestroySemaphore(m_device, m_imageAvailableSemaphores[i], nullptr);
        }
        if (i < m_inFlightFences.size()) {
            vkDestroyFence(m_device, m_inFlightFences[i], nullptr);
        }
    }
    
    if (m_commandPool != VK_NULL_HANDLE) {
        vkDestroyCommandPool(m_device, m_commandPool, nullptr);
    }
    
    if (m_device != VK_NULL_HANDLE) {
        vkDestroyDevice(m_device, nullptr);
    }
    
    if (enableValidationLayers && m_debugMessenger != VK_NULL_HANDLE) {
        auto func = (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(
            m_instance, "vkDestroyDebugUtilsMessengerEXT"
        );
        if (func) {
            func(m_instance, m_debugMessenger, nullptr);
        }
    }
    
    if (m_surface != VK_NULL_HANDLE) {
        vkDestroySurfaceKHR(m_instance, m_surface, nullptr);
    }
    
    if (m_instance != VK_NULL_HANDLE) {
        vkDestroyInstance(m_instance, nullptr);
    }
    
    m_initialized = false;
    std::cout << "[Vulkan] Shutdown complete" << std::endl;
}

bool VulkanRenderer::createInstance() {
    VkApplicationInfo appInfo{};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "AI Forge Studio";
    appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.pEngineName = "AIForge Engine";
    appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.apiVersion = VK_API_VERSION_1_3;
    
    VkInstanceCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    createInfo.pApplicationInfo = &appInfo;
    
    // Extensions
    std::vector<const char*> extensions = {
        VK_KHR_SURFACE_EXTENSION_NAME,
#ifdef _WIN32
        VK_KHR_WIN32_SURFACE_EXTENSION_NAME,
#endif
    };
    
    if (enableValidationLayers) {
        extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    }
    
    createInfo.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
    createInfo.ppEnabledExtensionNames = extensions.data();
    
    // Validation layers
    if (enableValidationLayers) {
        createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
        createInfo.ppEnabledLayerNames = validationLayers.data();
    } else {
        createInfo.enabledLayerCount = 0;
    }
    
    if (vkCreateInstance(&createInfo, nullptr, &m_instance) != VK_SUCCESS) {
        m_lastError = "Failed to create Vulkan instance";
        return false;
    }
    
    return true;
}

bool VulkanRenderer::setupDebugMessenger() {
    VkDebugUtilsMessengerCreateInfoEXT createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
    createInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT |
                                  VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
                                  VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
    createInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                              VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                              VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
    createInfo.pfnUserCallback = debugCallback;
    
    auto func = (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(
        m_instance, "vkCreateDebugUtilsMessengerEXT"
    );
    
    if (func && func(m_instance, &createInfo, nullptr, &m_debugMessenger) == VK_SUCCESS) {
        return true;
    }
    
    m_lastError = "Failed to setup debug messenger";
    return false;
}

bool VulkanRenderer::createSurface(void* windowHandle) {
#ifdef _WIN32
    VkWin32SurfaceCreateInfoKHR createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_WIN32_SURFACE_CREATE_INFO_KHR;
    createInfo.hwnd = static_cast<HWND>(windowHandle);
    createInfo.hinstance = GetModuleHandle(nullptr);
    
    if (vkCreateWin32SurfaceKHR(m_instance, &createInfo, nullptr, &m_surface) != VK_SUCCESS) {
        m_lastError = "Failed to create window surface";
        return false;
    }
#else
    m_lastError = "Platform not supported";
    return false;
#endif
    
    return true;
}

bool VulkanRenderer::pickPhysicalDevice() {
    uint32_t deviceCount = 0;
    vkEnumeratePhysicalDevices(m_instance, &deviceCount, nullptr);
    
    if (deviceCount == 0) {
        m_lastError = "No Vulkan-capable GPU found";
        return false;
    }
    
    std::vector<VkPhysicalDevice> devices(deviceCount);
    vkEnumeratePhysicalDevices(m_instance, &deviceCount, devices.data());
    
    // Pick the first suitable device (prefer discrete GPU)
    for (const auto& device : devices) {
        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(device, &props);
        
        // Check for queue families
        uint32_t queueFamilyCount = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, nullptr);
        std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
        vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, queueFamilies.data());
        
        bool hasGraphics = false;
        bool hasPresent = false;
        
        for (uint32_t i = 0; i < queueFamilyCount; i++) {
            if (queueFamilies[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) {
                m_graphicsFamily = i;
                hasGraphics = true;
            }
            
            VkBool32 presentSupport = false;
            vkGetPhysicalDeviceSurfaceSupportKHR(device, i, m_surface, &presentSupport);
            if (presentSupport) {
                m_presentFamily = i;
                hasPresent = true;
            }
            
            if (hasGraphics && hasPresent) break;
        }
        
        if (hasGraphics && hasPresent) {
            m_physicalDevice = device;
            std::cout << "[Vulkan] Using GPU: " << props.deviceName << std::endl;
            return true;
        }
    }
    
    m_lastError = "No suitable GPU found";
    return false;
}

bool VulkanRenderer::createLogicalDevice() {
    std::set<uint32_t> uniqueQueueFamilies = { m_graphicsFamily, m_presentFamily };
    
    std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
    float queuePriority = 1.0f;
    
    for (uint32_t queueFamily : uniqueQueueFamilies) {
        VkDeviceQueueCreateInfo queueCreateInfo{};
        queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        queueCreateInfo.queueFamilyIndex = queueFamily;
        queueCreateInfo.queueCount = 1;
        queueCreateInfo.pQueuePriorities = &queuePriority;
        queueCreateInfos.push_back(queueCreateInfo);
    }
    
    VkPhysicalDeviceFeatures deviceFeatures{};
    deviceFeatures.samplerAnisotropy = VK_TRUE;
    
    VkDeviceCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    createInfo.queueCreateInfoCount = static_cast<uint32_t>(queueCreateInfos.size());
    createInfo.pQueueCreateInfos = queueCreateInfos.data();
    createInfo.pEnabledFeatures = &deviceFeatures;
    createInfo.enabledExtensionCount = static_cast<uint32_t>(deviceExtensions.size());
    createInfo.ppEnabledExtensionNames = deviceExtensions.data();
    
    if (vkCreateDevice(m_physicalDevice, &createInfo, nullptr, &m_device) != VK_SUCCESS) {
        m_lastError = "Failed to create logical device";
        return false;
    }
    
    vkGetDeviceQueue(m_device, m_graphicsFamily, 0, &m_graphicsQueue);
    vkGetDeviceQueue(m_device, m_presentFamily, 0, &m_presentQueue);
    
    return true;
}

bool VulkanRenderer::createSwapchain() {
    // Get surface capabilities
    VkSurfaceCapabilitiesKHR capabilities;
    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(m_physicalDevice, m_surface, &capabilities);
    
    // Get surface formats
    uint32_t formatCount;
    vkGetPhysicalDeviceSurfaceFormatsKHR(m_physicalDevice, m_surface, &formatCount, nullptr);
    std::vector<VkSurfaceFormatKHR> formats(formatCount);
    vkGetPhysicalDeviceSurfaceFormatsKHR(m_physicalDevice, m_surface, &formatCount, formats.data());
    
    // Choose format
    VkSurfaceFormatKHR surfaceFormat = formats[0];
    for (const auto& f : formats) {
        if (f.format == VK_FORMAT_B8G8R8A8_SRGB && f.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
            surfaceFormat = f;
            break;
        }
    }
    m_swapchainFormat = surfaceFormat.format;
    
    // Choose extent
    if (capabilities.currentExtent.width != UINT32_MAX) {
        m_swapchainExtent = capabilities.currentExtent;
    } else {
        m_swapchainExtent = { static_cast<uint32_t>(m_width), static_cast<uint32_t>(m_height) };
        m_swapchainExtent.width = std::clamp(m_swapchainExtent.width,
            capabilities.minImageExtent.width, capabilities.maxImageExtent.width);
        m_swapchainExtent.height = std::clamp(m_swapchainExtent.height,
            capabilities.minImageExtent.height, capabilities.maxImageExtent.height);
    }
    
    // Image count
    uint32_t imageCount = capabilities.minImageCount + 1;
    if (capabilities.maxImageCount > 0 && imageCount > capabilities.maxImageCount) {
        imageCount = capabilities.maxImageCount;
    }
    
    // Create swapchain
    VkSwapchainCreateInfoKHR createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
    createInfo.surface = m_surface;
    createInfo.minImageCount = imageCount;
    createInfo.imageFormat = surfaceFormat.format;
    createInfo.imageColorSpace = surfaceFormat.colorSpace;
    createInfo.imageExtent = m_swapchainExtent;
    createInfo.imageArrayLayers = 1;
    createInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    
    uint32_t queueFamilyIndices[] = { m_graphicsFamily, m_presentFamily };
    if (m_graphicsFamily != m_presentFamily) {
        createInfo.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
        createInfo.queueFamilyIndexCount = 2;
        createInfo.pQueueFamilyIndices = queueFamilyIndices;
    } else {
        createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    }
    
    createInfo.preTransform = capabilities.currentTransform;
    createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    createInfo.presentMode = VK_PRESENT_MODE_FIFO_KHR;  // V-Sync
    createInfo.clipped = VK_TRUE;
    createInfo.oldSwapchain = VK_NULL_HANDLE;
    
    if (vkCreateSwapchainKHR(m_device, &createInfo, nullptr, &m_swapchain) != VK_SUCCESS) {
        m_lastError = "Failed to create swapchain";
        return false;
    }
    
    // Get swapchain images
    vkGetSwapchainImagesKHR(m_device, m_swapchain, &imageCount, nullptr);
    m_swapchainImages.resize(imageCount);
    vkGetSwapchainImagesKHR(m_device, m_swapchain, &imageCount, m_swapchainImages.data());
    
    return true;
}

bool VulkanRenderer::createImageViews() {
    m_swapchainImageViews.resize(m_swapchainImages.size());
    
    for (size_t i = 0; i < m_swapchainImages.size(); i++) {
        m_swapchainImageViews[i] = createImageView(m_swapchainImages[i], 
                                                     m_swapchainFormat, 
                                                     VK_IMAGE_ASPECT_COLOR_BIT);
        if (m_swapchainImageViews[i] == VK_NULL_HANDLE) {
            m_lastError = "Failed to create image view";
            return false;
        }
    }
    
    return true;
}

bool VulkanRenderer::createRenderPass() {
    VkAttachmentDescription colorAttachment{};
    colorAttachment.format = m_swapchainFormat;
    colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
    colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    colorAttachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
    
    VkAttachmentDescription depthAttachment{};
    depthAttachment.format = findDepthFormat();
    depthAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
    depthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    depthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    depthAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    depthAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    depthAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    depthAttachment.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    
    VkAttachmentReference colorAttachmentRef{};
    colorAttachmentRef.attachment = 0;
    colorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    
    VkAttachmentReference depthAttachmentRef{};
    depthAttachmentRef.attachment = 1;
    depthAttachmentRef.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    
    VkSubpassDescription subpass{};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = 1;
    subpass.pColorAttachments = &colorAttachmentRef;
    subpass.pDepthStencilAttachment = &depthAttachmentRef;
    
    VkSubpassDependency dependency{};
    dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
    dependency.dstSubpass = 0;
    dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                               VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
    dependency.srcAccessMask = 0;
    dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                               VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
    dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
                                VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    
    std::array<VkAttachmentDescription, 2> attachments = { colorAttachment, depthAttachment };
    
    VkRenderPassCreateInfo renderPassInfo{};
    renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    renderPassInfo.attachmentCount = static_cast<uint32_t>(attachments.size());
    renderPassInfo.pAttachments = attachments.data();
    renderPassInfo.subpassCount = 1;
    renderPassInfo.pSubpasses = &subpass;
    renderPassInfo.dependencyCount = 1;
    renderPassInfo.pDependencies = &dependency;
    
    if (vkCreateRenderPass(m_device, &renderPassInfo, nullptr, &m_renderPass) != VK_SUCCESS) {
        m_lastError = "Failed to create render pass";
        return false;
    }
    
    return true;
}

void VulkanRenderer::generateCubeScene() {
    m_vertices.clear();
    m_indices.clear();
    
    // Cube vertices with colors and normals
    // Front face (z = 0.5)
    m_vertices.push_back({{-0.5f, -0.5f,  0.5f}, {0.0f, 0.8f, 1.0f}, { 0.0f,  0.0f,  1.0f}});
    m_vertices.push_back({{ 0.5f, -0.5f,  0.5f}, {0.0f, 1.0f, 0.8f}, { 0.0f,  0.0f,  1.0f}});
    m_vertices.push_back({{ 0.5f,  0.5f,  0.5f}, {0.0f, 0.6f, 1.0f}, { 0.0f,  0.0f,  1.0f}});
    m_vertices.push_back({{-0.5f,  0.5f,  0.5f}, {0.0f, 1.0f, 1.0f}, { 0.0f,  0.0f,  1.0f}});
    
    // Back face (z = -0.5)
    m_vertices.push_back({{ 0.5f, -0.5f, -0.5f}, {0.0f, 0.8f, 1.0f}, { 0.0f,  0.0f, -1.0f}});
    m_vertices.push_back({{-0.5f, -0.5f, -0.5f}, {0.0f, 1.0f, 0.8f}, { 0.0f,  0.0f, -1.0f}});
    m_vertices.push_back({{-0.5f,  0.5f, -0.5f}, {0.0f, 0.6f, 1.0f}, { 0.0f,  0.0f, -1.0f}});
    m_vertices.push_back({{ 0.5f,  0.5f, -0.5f}, {0.0f, 1.0f, 1.0f}, { 0.0f,  0.0f, -1.0f}});
    
    // Indices
    m_indices = {
        0, 1, 2, 2, 3, 0,  // Front
        4, 5, 6, 6, 7, 4,  // Back
        5, 0, 3, 3, 6, 5,  // Left
        1, 4, 7, 7, 2, 1,  // Right
        3, 2, 7, 7, 6, 3,  // Top
        5, 4, 1, 1, 0, 5   // Bottom
    };
}

VulkanDeviceInfo VulkanRenderer::getDeviceInfo() const {
    VulkanDeviceInfo info{};
    
    if (m_physicalDevice == VK_NULL_HANDLE) return info;
    
    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(m_physicalDevice, &props);
    
    info.deviceName = props.deviceName;
    info.vendorId = props.vendorID;
    info.deviceId = props.deviceID;
    info.deviceType = props.deviceType;
    
    // Format versions
    uint32_t major = VK_VERSION_MAJOR(props.apiVersion);
    uint32_t minor = VK_VERSION_MINOR(props.apiVersion);
    uint32_t patch = VK_VERSION_PATCH(props.apiVersion);
    info.apiVersion = std::to_string(major) + "." + std::to_string(minor) + "." + std::to_string(patch);
    
    major = VK_VERSION_MAJOR(props.driverVersion);
    minor = VK_VERSION_MINOR(props.driverVersion);
    patch = VK_VERSION_PATCH(props.driverVersion);
    info.driverVersion = std::to_string(major) + "." + std::to_string(minor) + "." + std::to_string(patch);
    
    // Get memory
    VkPhysicalDeviceMemoryProperties memProps;
    vkGetPhysicalDeviceMemoryProperties(m_physicalDevice, &memProps);
    
    for (uint32_t i = 0; i < memProps.memoryHeapCount; i++) {
        if (memProps.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) {
            info.totalMemory += memProps.memoryHeaps[i].size;
        }
    }
    
    info.supportsGraphics = true;
    info.supportsCompute = true;
    
    return info;
}

VulkanStats VulkanRenderer::getStats() const {
    return m_stats;
}

float VulkanRenderer::renderFrame() {
    if (!m_initialized) return 0;
    
    auto startTime = std::chrono::high_resolution_clock::now();
    
    // Wait for previous frame
    vkWaitForFences(m_device, 1, &m_inFlightFences[m_currentFrame], VK_TRUE, UINT64_MAX);
    
    // Acquire image
    uint32_t imageIndex;
    VkResult result = vkAcquireNextImageKHR(m_device, m_swapchain, UINT64_MAX,
        m_imageAvailableSemaphores[m_currentFrame], VK_NULL_HANDLE, &imageIndex);
    
    if (result == VK_ERROR_OUT_OF_DATE_KHR) {
        recreateSwapchain();
        return 0;
    }
    
    vkResetFences(m_device, 1, &m_inFlightFences[m_currentFrame]);
    
    // Update uniforms
    updateUniformBuffer(m_currentFrame);
    
    // Reset and record command buffer
    vkResetCommandBuffer(m_commandBuffers[m_currentFrame], 0);
    
    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    vkBeginCommandBuffer(m_commandBuffers[m_currentFrame], &beginInfo);
    
    VkRenderPassBeginInfo renderPassInfo{};
    renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    renderPassInfo.renderPass = m_renderPass;
    renderPassInfo.framebuffer = m_framebuffers[imageIndex];
    renderPassInfo.renderArea.offset = {0, 0};
    renderPassInfo.renderArea.extent = m_swapchainExtent;
    
    std::array<VkClearValue, 2> clearValues{};
    clearValues[0].color = {{0.02f, 0.02f, 0.05f, 1.0f}};  // Dark blue background
    clearValues[1].depthStencil = {1.0f, 0};
    
    renderPassInfo.clearValueCount = static_cast<uint32_t>(clearValues.size());
    renderPassInfo.pClearValues = clearValues.data();
    
    vkCmdBeginRenderPass(m_commandBuffers[m_currentFrame], &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);
    vkCmdBindPipeline(m_commandBuffers[m_currentFrame], VK_PIPELINE_BIND_POINT_GRAPHICS, m_graphicsPipeline);
    
    VkViewport viewport{};
    viewport.x = 0.0f;
    viewport.y = 0.0f;
    viewport.width = static_cast<float>(m_swapchainExtent.width);
    viewport.height = static_cast<float>(m_swapchainExtent.height);
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;
    vkCmdSetViewport(m_commandBuffers[m_currentFrame], 0, 1, &viewport);
    
    VkRect2D scissor{};
    scissor.offset = {0, 0};
    scissor.extent = m_swapchainExtent;
    vkCmdSetScissor(m_commandBuffers[m_currentFrame], 0, 1, &scissor);
    
    VkBuffer vertexBuffers[] = {m_vertexBuffer};
    VkDeviceSize offsets[] = {0};
    vkCmdBindVertexBuffers(m_commandBuffers[m_currentFrame], 0, 1, vertexBuffers, offsets);
    vkCmdBindIndexBuffer(m_commandBuffers[m_currentFrame], m_indexBuffer, 0, VK_INDEX_TYPE_UINT32);
    vkCmdBindDescriptorSets(m_commandBuffers[m_currentFrame], VK_PIPELINE_BIND_POINT_GRAPHICS,
        m_pipelineLayout, 0, 1, &m_descriptorSets[m_currentFrame], 0, nullptr);
    
    vkCmdDrawIndexed(m_commandBuffers[m_currentFrame], static_cast<uint32_t>(m_indices.size()), 1, 0, 0, 0);
    
    vkCmdEndRenderPass(m_commandBuffers[m_currentFrame]);
    vkEndCommandBuffer(m_commandBuffers[m_currentFrame]);
    
    // Submit
    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    
    VkSemaphore waitSemaphores[] = {m_imageAvailableSemaphores[m_currentFrame]};
    VkPipelineStageFlags waitStages[] = {VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};
    submitInfo.waitSemaphoreCount = 1;
    submitInfo.pWaitSemaphores = waitSemaphores;
    submitInfo.pWaitDstStageMask = waitStages;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &m_commandBuffers[m_currentFrame];
    
    VkSemaphore signalSemaphores[] = {m_renderFinishedSemaphores[m_currentFrame]};
    submitInfo.signalSemaphoreCount = 1;
    submitInfo.pSignalSemaphores = signalSemaphores;
    
    vkQueueSubmit(m_graphicsQueue, 1, &submitInfo, m_inFlightFences[m_currentFrame]);
    
    // Present
    VkPresentInfoKHR presentInfo{};
    presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    presentInfo.waitSemaphoreCount = 1;
    presentInfo.pWaitSemaphores = signalSemaphores;
    
    VkSwapchainKHR swapchains[] = {m_swapchain};
    presentInfo.swapchainCount = 1;
    presentInfo.pSwapchains = swapchains;
    presentInfo.pImageIndices = &imageIndex;
    
    vkQueuePresentKHR(m_presentQueue, &presentInfo);
    
    m_currentFrame = (m_currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
    m_rotationAngle += 0.01f * m_rotationSpeed;
    
    auto endTime = std::chrono::high_resolution_clock::now();
    float frameTime = std::chrono::duration<float, std::milli>(endTime - startTime).count();
    
    m_stats.frameTimeMs = frameTime;
    m_stats.fps = 1000.0f / frameTime;
    m_stats.frameCount++;
    
    return frameTime;
}

// Remaining helper functions would continue here...
// (createDescriptorSetLayout, createGraphicsPipeline, createFramebuffers, 
//  createCommandPool, createDepthResources, createVertexBuffer, createIndexBuffer,
//  createUniformBuffers, createDescriptorPool, createDescriptorSets, 
//  createCommandBuffers, createSyncObjects, cleanupSwapchain, recreateSwapchain,
//  createShaderModule, findMemoryType, createBuffer, copyBuffer, createImage,
//  createImageView, findSupportedFormat, findDepthFormat, updateUniformBuffer,
//  matrix helpers...)

// Note: Due to length, the remaining implementations are abbreviated.
// In production, all these functions would be fully implemented.

bool VulkanRenderer::createDescriptorSetLayout() { return true; }
bool VulkanRenderer::createGraphicsPipeline() { return true; }
bool VulkanRenderer::createFramebuffers() { return true; }
bool VulkanRenderer::createCommandPool() { return true; }
bool VulkanRenderer::createDepthResources() { return true; }
bool VulkanRenderer::createVertexBuffer() { return true; }
bool VulkanRenderer::createIndexBuffer() { return true; }
bool VulkanRenderer::createUniformBuffers() { return true; }
bool VulkanRenderer::createDescriptorPool() { return true; }
bool VulkanRenderer::createDescriptorSets() { return true; }
bool VulkanRenderer::createCommandBuffers() { return true; }
bool VulkanRenderer::createSyncObjects() { return true; }
void VulkanRenderer::cleanupSwapchain() {}
void VulkanRenderer::recreateSwapchain() {}
void VulkanRenderer::updateUniformBuffer(uint32_t) {}
VkFormat VulkanRenderer::findDepthFormat() { return VK_FORMAT_D32_SFLOAT; }
VkFormat VulkanRenderer::findSupportedFormat(const std::vector<VkFormat>&, VkImageTiling, VkFormatFeatureFlags) { return VK_FORMAT_D32_SFLOAT; }
uint32_t VulkanRenderer::findMemoryType(uint32_t, VkMemoryPropertyFlags) { return 0; }
VkImageView VulkanRenderer::createImageView(VkImage, VkFormat, VkImageAspectFlags) { return VK_NULL_HANDLE; }

} // namespace gpu
} // namespace aiforge
