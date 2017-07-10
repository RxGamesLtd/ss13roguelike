#pragma once

#include "glfw_wrapper.hpp"
#include "vulkan/vulkan.hpp"

#include <chrono>

class Renderer
{
public:
    Renderer(GLFWwindow* window, const std::vector<const char*>& instanceExtensions,
             const std::vector<const char*>& requestedDeviceExtensions);

    bool isValid() { return _isInited; }

private:
    void _initInstance(const std::vector<const char*>& instanceExtensions);
    void _initSurface(GLFWwindow* window);
    void _initDevice(const std::vector<const char*>& requestedDeviceExtensions);
    void _initSwapchain(vk::Extent2D desiredExtent);
    void _initImageViews();
    void _initRenderPass();
    void _initPipeline();
    void _initFramebuffers();

    bool _fillQueueFamilies(vk::PhysicalDevice& gpu);
    bool _isDeviceCompatible(const vk::PhysicalDevice& device,
                             const std::vector<const char*>& requestedDeviceExtensions);

public:
    // Instance block
    vk::UniqueInstance _instance;
    vk::PhysicalDevice _gpu;
    vk::UniqueDevice _device;
    vk::UniqueSurfaceKHR _surface;
    vk::Queue _queue;
    vk::Queue _presentQueue;
    std::vector<vk::UniqueFramebuffer> _framebuffers;

    // pipeline
    vk::UniquePipelineLayout _pipelineLayout;
    vk::UniquePipelineCache _pipelineCache;
    vk::UniqueRenderPass _renderPass;
    vk::UniquePipeline _graphicsPipeline;
    // Swapchain block
    vk::UniqueSwapchainKHR _swapchain;
    std::vector<vk::Image> _swapchainImages;
    std::vector<vk::UniqueImageView> _swapChainImageViews;
    vk::Format _swapchainFormat;
    vk::Extent2D _swapchainExtent;

    // Props
    uint32_t _graphicsFamilyIdx = 0;
    uint32_t _presentFamilyIdx = 0;

    // init
    bool _isInited;
    std::chrono::time_point<std::chrono::high_resolution_clock> _frameStartTime;
};
