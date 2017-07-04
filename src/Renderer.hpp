#pragma once

#include "vulkan/vulkan.hpp"
#include "GLFW/glfw3.h"

class Renderer
{
  public:
    Renderer(GLFWwindow* window, const std::vector<const char*> extensions);

  private:
    void _initInstance(const std::vector<const char*> extensions);
    void _initSurface(GLFWwindow* window);
    void _initDevice();

    void _getQueueFamilies(const vk::PhysicalDevice* gpu);

  public:
    // Handles block

    vk::UniqueInstance _instance;
    vk::UniqueDevice _device;
    vk::UniqueSurfaceKHR _surface;
    vk::Queue _queue;
    vk::Queue _presentQueue;

    // Props
    uint32_t _graphicsFamilyIdx = 0;
    uint32_t _presentFamilyIdx = 0;
};
