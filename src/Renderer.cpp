#include "renderer.hpp"

#include "GLFW/glfw3.h"
#include "vulkan/vulkan.hpp"
#include <assert.h>
#include <set>
#include <iostream>

Renderer::Renderer(GLFWwindow* window, const std::vector<const char*>& extensions)
{
    _initInstance(extensions);
    _initSurface(window);
    _initDevice();
}

void Renderer::_initInstance(const std::vector<const char*> extensions)
{
    // Use validation layers if this is a debug build
    std::vector<const char*> layers;
#if defined(_DEBUG)
    const auto instanceLayerProps = vk::enumerateInstanceLayerProperties();
    for(const auto& lp : instanceLayerProps)
    {
        const std::string lpName(lp.layerName);

        std::cout << "Instance Layer: " << lpName << "\t|\t" << lp.description << "\n";

        if(lpName.find("VK_LAYER_LUNARG_standard_validation") != std::string::npos)
        {
            layers.push_back("VK_LAYER_LUNARG_standard_validation");
        }
    }
    std::cout << std::endl;
#endif

    // VkApplicationInfo allows the programmer to specifiy some basic information
    // about the program, which can be useful for layers and tools to provide
    // more debug information.
    auto appInfo = vk::ApplicationInfo() //
                     .setPApplicationName("SS13 rogue-like") //
                     .setApplicationVersion(VK_MAKE_VERSION(1, 0, 0)) //
                     .setPEngineName("LunarG SDK") //
                     .setEngineVersion(1) //
                     .setApiVersion(VK_API_VERSION_1_0); //

    // VkInstanceCreateInfo is where the programmer specifies the layers and/or
    // extensions that are needed. For now, none are enabled.
    auto instInfo = vk::InstanceCreateInfo() //
                      .setFlags(vk::InstanceCreateFlags()) //
                      .setPApplicationInfo(&appInfo) //
                      .setEnabledExtensionCount(static_cast<uint32_t>(extensions.size())) //
                      .setPpEnabledExtensionNames(extensions.data()) //
                      .setEnabledLayerCount(static_cast<uint32_t>(layers.size())) //
                      .setPpEnabledLayerNames(layers.data()); //

    // Create the Vulkan instance.
    try
    {
        _instance = vk::createInstanceUnique(instInfo);
    }
    catch(std::exception e)
    {
        std::cout << "Could not create a Vulkan instance: " << e.what() << std::endl;
        std::exit(1);
    }
}

void Renderer::_initSurface(GLFWwindow* window)
{
    VkSurfaceKHR surf{};
    if(glfwCreateWindowSurface(VkInstance(_instance.get()), window, nullptr, &surf) != VK_SUCCESS)
    {
        std::cout << "Error creating surface" << std::endl;
        std::exit(-1);
    }
    vk::SurfaceKHRDeleter deleter(_instance.get(), nullptr);
    _surface = vk::UniqueSurfaceKHR(surf, deleter);
}

void Renderer::_getQueueFamilies(vk::PhysicalDevice& gpu)
{
    const auto queues = gpu.getQueueFamilyProperties();
    {
        const auto q = std::find_if(queues.begin(), queues.end(), [&](const auto& d) {
            //const auto curIndex = static_cast<uint32_t>(&d - &queues[0]);
            // bool presentSupport =
            //   glfwGetPhysicalDevicePresentationSupport(VkInstance(_instance.get()), VkPhysicalDevice(gpu), curIndex);
            // bool surfaceSupport = !_surface || gpu->getSurfaceSupportKHR(curIndex, _surface.get());
            return d.queueFlags & vk::QueueFlagBits::eGraphics;
        });

        if(q == queues.end())
        {
            std::cout << "Could not find a appropriate queue!" << std::endl;
            std::exit(1);
        }
        _graphicsFamilyIdx = static_cast<uint32_t>(q - queues.begin());
    }
    {
        const auto q = std::find_if(queues.begin(), queues.end(), [&](const auto& d) {
            const auto curIndex = static_cast<uint32_t>(&d - &queues[0]);
            bool presentSupport =
              glfwGetPhysicalDevicePresentationSupport(VkInstance(_instance.get()), VkPhysicalDevice(gpu), curIndex);
            bool surfaceSupport = !_surface || gpu.getSurfaceSupportKHR(curIndex, _surface.get());
            return surfaceSupport && presentSupport;
        });

        if(q == queues.end())
        {
            std::cout << "Could not find a appropriate queue!" << std::endl;
            std::exit(1);
        }
        _presentFamilyIdx = static_cast<uint32_t>(q - queues.begin());
    }
}

void Renderer::_initDevice()
{
    auto gpus = _instance->enumeratePhysicalDevices();

    // Find physical device
    vk::PhysicalDevice* gpu = nullptr;
    {
        for(auto& device : gpus)
        {
            const auto props = device.getProperties();

            std::cout << "Device: " << props.deviceName << "\n";
            std::cout << "\tType:" << static_cast<uint32_t>(props.deviceType) << "\n";
            std::cout << "\tDriver:" << props.driverVersion << "\n";
            std::cout << std::endl;
            if(props.deviceType == vk::PhysicalDeviceType::eDiscreteGpu)
            {
                gpu = &device;
            }
        }
        if(!gpu)
        {
            std::cout << "Could not find a physical device!" << std::endl;
            std::exit(1);
        }
    }

    // Find appropriate queues
    _getQueueFamilies(*gpu);

    std::vector<vk::DeviceQueueCreateInfo> queueCreateInfos;
    std::set<uint32_t> uniqueQueueFamilies = { _presentFamilyIdx, _graphicsFamilyIdx };

    float queuePriority = 1.0f;
    for(const auto& queueFamily : uniqueQueueFamilies)
    {
        auto devQueueCreateInfo = vk::DeviceQueueCreateInfo() //
                                    .setQueueFamilyIndex(queueFamily) //
                                    .setQueueCount(1) //
                                    .setPQueuePriorities(&queuePriority);
        queueCreateInfos.push_back(devQueueCreateInfo);
    }

    auto devCreateInfo = vk::DeviceCreateInfo() //
                           .setQueueCreateInfoCount(1u) //
                           .setPQueueCreateInfos(queueCreateInfos.data());

    try
    {
        _device = gpu->createDeviceUnique(devCreateInfo);
    }
    catch(std::exception e)
    {
        std::cout << "Could not create a Vulkan device: " << e.what() << std::endl;
        std::exit(1);
    }

    try
    {
        _queue = _device->getQueue(_graphicsFamilyIdx, 0u);
        _presentQueue = _device->getQueue(_presentFamilyIdx, 0u);
    }
    catch(std::exception e)
    {
        std::cout << "Could not get a Vulkan queue: " << e.what() << std::endl;
        std::exit(1);
    }
}
