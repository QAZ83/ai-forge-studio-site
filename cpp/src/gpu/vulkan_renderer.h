/**
 * AI Forge Studio - Vulkan Renderer Module
 * Author: M.3R3
 * 
 * Minimal Vulkan rendering pipeline for GPU visualization.
 */

#pragma once

#include <vulkan/vulkan.h>
#include <vector>
#include <string>
#include <memory>
#include <array>
#include <functional>

// Forward declarations
class QWidget;
class QWindow;

namespace aiforge {
namespace gpu {

/**
 * Vulkan device information
 */
struct VulkanDeviceInfo {
    std::string deviceName;
    std::string driverVersion;
    std::string apiVersion;
    uint32_t vendorId;
    uint32_t deviceId;
    VkPhysicalDeviceType deviceType;
    size_t totalMemory;
    bool supportsCompute;
    bool supportsGraphics;
};

/**
 * Vulkan renderer statistics
 */
struct VulkanStats {
    float frameTimeMs;
    float fps;
    uint64_t frameCount;
    size_t memoryUsed;
    size_t memoryBudget;
};

/**
 * Simple vertex structure
 */
struct Vertex {
    float position[3];
    float color[3];
    float normal[3];
    
    static VkVertexInputBindingDescription getBindingDescription();
    static std::array<VkVertexInputAttributeDescription, 3> getAttributeDescriptions();
};

/**
 * Uniform buffer object for transforms
 */
struct UniformBufferObject {
    alignas(16) float model[16];
    alignas(16) float view[16];
    alignas(16) float projection[16];
    alignas(16) float lightPos[4];
    alignas(16) float viewPos[4];
    float time;
    float padding[3];
};

/**
 * Vulkan Renderer
 * 
 * Provides a minimal Vulkan rendering pipeline for visualization.
 */
class VulkanRenderer {
public:
    VulkanRenderer();
    ~VulkanRenderer();

    // Disable copy
    VulkanRenderer(const VulkanRenderer&) = delete;
    VulkanRenderer& operator=(const VulkanRenderer&) = delete;

    /**
     * Initialize Vulkan with a native window handle
     * @param windowHandle Native window handle (HWND on Windows)
     * @param width Window width
     * @param height Window height
     * @return true if initialization successful
     */
    bool initialize(void* windowHandle, int width, int height);

    /**
     * Initialize for Qt widget rendering
     * @param window Qt window
     * @return true if initialization successful
     */
    bool initializeForQt(QWindow* window);

    /**
     * Shutdown Vulkan and cleanup resources
     */
    void shutdown();

    /**
     * Check if renderer is ready
     */
    bool isReady() const { return m_initialized; }

    /**
     * Resize the rendering surface
     * @param width New width
     * @param height New height
     */
    void resize(int width, int height);

    /**
     * Render a single frame
     * @return Frame time in milliseconds
     */
    float renderFrame();

    /**
     * Get device information
     */
    VulkanDeviceInfo getDeviceInfo() const;

    /**
     * Get rendering statistics
     */
    VulkanStats getStats() const;

    /**
     * Get last error message
     */
    std::string getLastError() const { return m_lastError; }

    /**
     * Set rotation speed for the demo scene
     */
    void setRotationSpeed(float speed) { m_rotationSpeed = speed; }

    /**
     * Set demo scene type
     * 0 = Rotating cube
     * 1 = GPU stress test (many triangles)
     * 2 = Particle system
     */
    void setDemoScene(int scene);

    /**
     * Get list of available GPUs
     */
    static std::vector<VulkanDeviceInfo> enumerateGPUs();

private:
    bool m_initialized;
    std::string m_lastError;
    
    // Window properties
    int m_width;
    int m_height;
    void* m_windowHandle;
    
    // Vulkan core objects
    VkInstance m_instance;
    VkPhysicalDevice m_physicalDevice;
    VkDevice m_device;
    VkSurfaceKHR m_surface;
    VkSwapchainKHR m_swapchain;
    VkRenderPass m_renderPass;
    VkPipelineLayout m_pipelineLayout;
    VkPipeline m_graphicsPipeline;
    VkCommandPool m_commandPool;
    VkDescriptorPool m_descriptorPool;
    VkDescriptorSetLayout m_descriptorSetLayout;
    
    // Queues
    VkQueue m_graphicsQueue;
    VkQueue m_presentQueue;
    uint32_t m_graphicsFamily;
    uint32_t m_presentFamily;
    
    // Swapchain
    std::vector<VkImage> m_swapchainImages;
    std::vector<VkImageView> m_swapchainImageViews;
    std::vector<VkFramebuffer> m_framebuffers;
    VkFormat m_swapchainFormat;
    VkExtent2D m_swapchainExtent;
    
    // Depth buffer
    VkImage m_depthImage;
    VkDeviceMemory m_depthImageMemory;
    VkImageView m_depthImageView;
    
    // Command buffers
    std::vector<VkCommandBuffer> m_commandBuffers;
    
    // Sync objects
    std::vector<VkSemaphore> m_imageAvailableSemaphores;
    std::vector<VkSemaphore> m_renderFinishedSemaphores;
    std::vector<VkFence> m_inFlightFences;
    uint32_t m_currentFrame;
    static constexpr int MAX_FRAMES_IN_FLIGHT = 2;
    
    // Buffers
    VkBuffer m_vertexBuffer;
    VkDeviceMemory m_vertexBufferMemory;
    VkBuffer m_indexBuffer;
    VkDeviceMemory m_indexBufferMemory;
    std::vector<VkBuffer> m_uniformBuffers;
    std::vector<VkDeviceMemory> m_uniformBuffersMemory;
    std::vector<void*> m_uniformBuffersMapped;
    std::vector<VkDescriptorSet> m_descriptorSets;
    
    // Scene data
    std::vector<Vertex> m_vertices;
    std::vector<uint32_t> m_indices;
    float m_rotationAngle;
    float m_rotationSpeed;
    int m_demoScene;
    
    // Statistics
    VulkanStats m_stats;
    
    // Debug
    VkDebugUtilsMessengerEXT m_debugMessenger;
    
    // Initialization helpers
    bool createInstance();
    bool setupDebugMessenger();
    bool createSurface(void* windowHandle);
    bool pickPhysicalDevice();
    bool createLogicalDevice();
    bool createSwapchain();
    bool createImageViews();
    bool createRenderPass();
    bool createDescriptorSetLayout();
    bool createGraphicsPipeline();
    bool createFramebuffers();
    bool createCommandPool();
    bool createDepthResources();
    bool createVertexBuffer();
    bool createIndexBuffer();
    bool createUniformBuffers();
    bool createDescriptorPool();
    bool createDescriptorSets();
    bool createCommandBuffers();
    bool createSyncObjects();
    
    // Cleanup helpers
    void cleanupSwapchain();
    void recreateSwapchain();
    
    // Utility
    VkShaderModule createShaderModule(const std::vector<uint32_t>& code);
    uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties);
    void createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, 
                      VkMemoryPropertyFlags properties, VkBuffer& buffer, 
                      VkDeviceMemory& bufferMemory);
    void copyBuffer(VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size);
    void createImage(uint32_t width, uint32_t height, VkFormat format, 
                     VkImageTiling tiling, VkImageUsageFlags usage,
                     VkMemoryPropertyFlags properties, VkImage& image, 
                     VkDeviceMemory& imageMemory);
    VkImageView createImageView(VkImage image, VkFormat format, VkImageAspectFlags aspectFlags);
    VkFormat findSupportedFormat(const std::vector<VkFormat>& candidates,
                                  VkImageTiling tiling, VkFormatFeatureFlags features);
    VkFormat findDepthFormat();
    
    // Scene generation
    void generateCubeScene();
    void generateStressTestScene();
    void generateParticleScene();
    void updateUniformBuffer(uint32_t currentImage);
    
    // Matrix helpers
    static void mat4Identity(float* m);
    static void mat4Multiply(float* result, const float* a, const float* b);
    static void mat4RotateY(float* m, float angle);
    static void mat4RotateX(float* m, float angle);
    static void mat4Translate(float* m, float x, float y, float z);
    static void mat4Perspective(float* m, float fov, float aspect, float near, float far);
    static void mat4LookAt(float* m, const float* eye, const float* center, const float* up);
};

} // namespace gpu
} // namespace aiforge
