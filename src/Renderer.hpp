#pragma once

#include "vulkan/vulkan.hpp"

#include <chrono>

struct GLFWwindow;

class Material;

class Renderer
{
public:
    Renderer(GLFWwindow* window, const std::vector<const char*>& instanceExtensions,
             const std::vector<const char*>& requestedDeviceExtensions);

    bool isValid() { return m_isInited; }

    vk::Device getDevice() const { return m_device.get(); }

    void prepairFor(const Material& mat);
    void beginRender();
    void endRender();
    void present();

    void waitForIdle();

private:
    void initInstance(const std::vector<const char*>& instanceExtensions);
    void initSurface(GLFWwindow* window);
    void initDevice(const std::vector<const char*>& requestedDeviceExtensions);
    void initSwapchain(vk::Extent2D desiredExtent);
    void initImageViews();
    void initPipelineCache();
    void initCommandPool();
    void initSyncObjects();

    void initRenderPass();
    void initPipeline(const Material& mat);
    void initFramebuffers();
    void initCommandBuffers();

protected:
    bool fillQueueFamilies(vk::PhysicalDevice& gpu);
    bool isDeviceCompatible(const vk::PhysicalDevice& device,
                             const std::vector<const char*>& requestedDeviceExtensions);

public:
    // Instance block
    vk::UniqueInstance m_instance;
    vk::PhysicalDevice m_gpu;
    vk::UniqueDevice m_device;
    vk::UniqueSurfaceKHR m_surface;
    vk::Queue m_queue;
    vk::Queue m_presentQueue;
    std::vector<vk::UniqueFramebuffer> m_framebuffers;

    // pipeline
    vk::UniquePipelineCache m_pipelineCache;
    // render pipeline
    vk::UniquePipelineLayout m_pipelineLayout;
    vk::UniqueRenderPass m_renderPass;
    vk::UniquePipeline m_graphicsPipeline;
    // Swapchain block
    vk::UniqueSwapchainKHR m_swapchain;
    std::vector<vk::Image> m_swapchainImages;
    std::vector<vk::UniqueImageView> m_swapChainImageViews;
    vk::Format m_swapchainFormat;
    vk::Extent2D m_swapchainExtent;

    //Commands
    vk::UniqueCommandPool m_commandPool;
    std::vector<vk::UniqueCommandBuffer> m_commandBuffers;

    //Synchronization
    std::vector<vk::UniqueSemaphore> m_renderFinishedSemaphores;
    std::vector<vk::UniqueSemaphore> m_imageAvailableSemaphores;
    std::vector<vk::UniqueFence> m_inFlightFences;

    // Props
    uint32_t m_graphicsFamilyIdx = 0;
    uint32_t m_presentFamilyIdx = 0;

    // Frame handling
    uint32_t m_currentImageIndex = -1;
    uint32_t m_currentFrameIndex = 0;

    // init
    bool m_isInited;
    std::chrono::time_point<std::chrono::high_resolution_clock> m_frameStartTime;
};
