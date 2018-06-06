#pragma once

#include "vulkan/vulkan.hpp"

#include <chrono>

struct GLFWwindow;

class Renderer
{
public:
    Renderer(GLFWwindow* window, const std::vector<const char*>& instanceExtensions,
             const std::vector<const char*>& requestedDeviceExtensions);

    bool isValid() { return m_isInited; }

private:
    void initInstance(const std::vector<const char*>& instanceExtensions);
    void initSurface(GLFWwindow* window);
    void initDevice(const std::vector<const char*>& requestedDeviceExtensions);
    void initSwapchain(vk::Extent2D desiredExtent);
    void initImageViews();
    void initRenderPass();
    void initPipeline();
    void initFramebuffers();
    void initCommandPool();
    void initCommandBuffers();
    void initSemaphores();

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
    vk::UniquePipelineLayout m_pipelineLayout;
    vk::UniquePipelineCache m_pipelineCache;
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

    //Synchronzation
    vk::UniqueSemaphore m_renderFinishedSemaphore;
    vk::UniqueSemaphore m_imageAvailableSemaphore;

    // Props
    uint32_t m_graphicsFamilyIdx = 0;
    uint32_t m_presentFamilyIdx = 0;

    // init
    bool m_isInited;
    std::chrono::time_point<std::chrono::high_resolution_clock> m_frameStartTime;
};
