#include "targetver.hpp"
#include "renderer.hpp"

#include "GLFW/glfw3.h"
#include "glm/glm.hpp"
#include "vulkan/vulkan.hpp"

#include <iostream>

inline void testStuff(const Renderer& r)
{
    const auto cpci = vk::CommandPoolCreateInfo() //
                        .setQueueFamilyIndex(r._graphicsFamilyIdx) //
                        .setFlags(vk::CommandPoolCreateFlags(vk::CommandPoolCreateFlagBits::eResetCommandBuffer));

    const auto commandPool = r._device->createCommandPoolUnique(cpci);

    // create buffers
    const auto cbci = vk::CommandBufferAllocateInfo() //
                        .setCommandPool(commandPool.get()) //
                        .setCommandBufferCount(1) //
                        .setLevel(vk::CommandBufferLevel::ePrimary);

    const auto commandBuffers = r._device->allocateCommandBuffersUnique(cbci);

    const auto cbbi = vk::CommandBufferBeginInfo();
    //.setFlags(vk::CommandBufferUsageFlags(vk::CommandBufferUsageFlagBits::eOneTimeSubmit));

    commandBuffers[0]->begin(cbbi);

    commandBuffers[0]->end();

    const auto fci = vk::FenceCreateInfo() //
                       .setFlags(vk::FenceCreateFlags());

    std::vector<vk::CommandBuffer> cbs(commandBuffers.size());
    std::transform(commandBuffers.begin(), commandBuffers.end(), cbs.begin(), [](const auto& t) { return t.get(); });
    vk::UniqueFence f = r._device->createFenceUnique(fci);

    const auto submitInfo = vk::SubmitInfo() //
                              .setCommandBufferCount(static_cast<uint32_t>(cbs.size())) //
                              .setPCommandBuffers(cbs.data());

    r._queue.submit(submitInfo, f.get());

    r._device->waitForFences(std::vector<vk::Fence>{ f.get() }, true, UINT64_MAX);
}

inline std::vector<const char*> getRequiredExtensions()
{
    std::vector<const char*> extensions;

    unsigned int glfwExtensionCount = 0;
    const char** glfwExtensions;
    glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

    extensions.assign(glfwExtensions, glfwExtensions + glfwExtensionCount);

#ifdef _DEBUG
    extensions.push_back(VK_EXT_DEBUG_REPORT_EXTENSION_NAME);
#endif

    return extensions;
}

inline void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
    (void)(scancode);
    (void)(mods);
    if(key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
    {
        glfwSetWindowShouldClose(window, GLFW_TRUE);
    }
}

int main()
{
    if(!glfwInit())
    {
        std::cout << "Error on init GLFW" << std::endl;
        std::exit(-1);
    }
    try
    {
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
        glfwWindowHint(GLFW_MAXIMIZED, GLFW_TRUE);
        // glfwWindowHint(GLFW_DECORATED, GLFW_FALSE);

        // glfwWindowHint(GLFW_RESIZABLE, 0);
        // glfwWindowHint(GLFW_RESIZABLE, 0);
        auto* window = glfwCreateWindow(800, 600, "TestApp", nullptr, nullptr);

        glfwSetKeyCallback(window, key_callback);

        Renderer render(window, getRequiredExtensions());

        while(!glfwWindowShouldClose(window))
        {
            glfwPollEvents();
        }
    }
    catch(std::runtime_error e)
    {
        std::cout << e.what() << std::endl;
    }
    glfwTerminate();
    return 0;
}
