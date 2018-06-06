#include "config.hpp"
#include "renderer.hpp"
#include "targetver.hpp"

#define GLFW_INCLUDE_VULKAN
#include "GLFW/glfw3.h"
#include "glm/glm.hpp"
#include "vulkan/vulkan.hpp"

#include <iostream>

static void testStuff(const Renderer& r)
{
    auto imageIndex = r.m_device->acquireNextImageKHR(r.m_swapchain.get(), UINT64_MAX, r.m_imageAvailableSemaphore.get(), nullptr);
    for (int i = 0; i < r.m_commandBuffers.size(); ++i)
    {
        const auto cbbi = vk::CommandBufferBeginInfo()
                          .setFlags(vk::CommandBufferUsageFlagBits::eSimultaneousUse);

        r.m_commandBuffers[i]->begin(cbbi);

        const auto clearValue = vk::ClearValue(vk::ClearColorValue(std::array<float, 4>{0.f, 0.f, 0.f, 0.f}));
        const auto rpbi = vk::RenderPassBeginInfo()
                          .setRenderPass(r.m_renderPass.get())
                          .setFramebuffer(r.m_framebuffers[i].get())
                          .setRenderArea(vk::Rect2D({ 0, 0 }, r.m_swapchainExtent))
                          .setClearValueCount(1)
                          .setPClearValues(&clearValue);

        r.m_commandBuffers[i]->beginRenderPass(rpbi, vk::SubpassContents::eInline);

        r.m_commandBuffers[i]->bindPipeline(vk::PipelineBindPoint::eGraphics, r.m_graphicsPipeline.get());

        r.m_commandBuffers[i]->draw(3, 1, 0, 0);

        r.m_commandBuffers[i]->endRenderPass();
        r.m_commandBuffers[i]->end();
    }

    std::vector<vk::CommandBuffer> cbs(r.m_commandBuffers.size());
    std::transform(r.m_commandBuffers.begin(), r.m_commandBuffers.end(), cbs.begin(), [](const auto& t)
    {
        return t.get();
    });

    std::vector<vk::PipelineStageFlags> waitStages = {vk::PipelineStageFlagBits::eColorAttachmentOutput};

    const auto submitInfo = vk::SubmitInfo()                                          //
                            .setCommandBufferCount(static_cast<uint32_t>(cbs.size())) //
                            .setPCommandBuffers(cbs.data())
                            .setPWaitSemaphores(&r.m_imageAvailableSemaphore.get())
                            .setWaitSemaphoreCount(1)
                            .setPWaitDstStageMask(waitStages.data())
                            .setPSignalSemaphores(&r.m_renderFinishedSemaphore.get())
                            .setSignalSemaphoreCount(1);

    r.m_queue.submit(submitInfo, nullptr);

    const auto presentInfo = vk::PresentInfoKHR() //
                                .setSwapchainCount(1) //
                                .setPSwapchains(&r.m_swapchain.get()) //
                                .setPImageIndices(&imageIndex.value);
    r.m_presentQueue.presentKHR(presentInfo);
}

inline std::vector<const char*>getRequiredInstanceExtensions()
{
    std::vector<const char*> extensions;

    unsigned int glfwExtensionCount = 0;
    const char** glfwExtensions     = nullptr;
    glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

    extensions.assign(glfwExtensions, glfwExtensions + glfwExtensionCount);

    if constexpr (Config::isDebug)
    {
        extensions.push_back(VK_EXT_DEBUG_REPORT_EXTENSION_NAME);
    }

    return extensions;
}

inline std::vector<const char*>getRequiredDeviceExtensions()
{
    std::vector<const char*> extensions;

    extensions.push_back(VK_KHR_SWAPCHAIN_EXTENSION_NAME);

    return extensions;
}

inline void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
    (void)(scancode);
    (void)(mods);

    if ((key == GLFW_KEY_ESCAPE) && (action == GLFW_PRESS))
    {
        glfwSetWindowShouldClose(window, GLFW_TRUE);
    }
}

int main()
{
    if (!glfwInit())
    {
        std::cout << "Error on init GLFW" << std::endl;
        std::exit(-1);
    }
    try
    {
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        glfwWindowHint(GLFW_RESIZABLE,  GLFW_FALSE);

        // glfwWindowHint(GLFW_MAXIMIZED, GLFW_TRUE);
        // glfwWindowHint(GLFW_DECORATED, GLFW_FALSE);

        // glfwWindowHint(GLFW_RESIZABLE, 0);
        // glfwWindowHint(GLFW_RESIZABLE, 0);
        auto window = glfwCreateWindow(1024, 768, "TestApp:Initilizing", nullptr, nullptr);

        glfwSetKeyCallback(window, keyCallback);

        auto instanceExtensions = getRequiredInstanceExtensions();
        auto deviceExtensions   = getRequiredDeviceExtensions();

        // init renderer
        Renderer renderer(window, instanceExtensions, deviceExtensions);

        // init done
        glfwSetWindowTitle(window, "TestApp");

        while (!glfwWindowShouldClose(window))
        {
            glfwPollEvents();
            renderer.m_frameStartTime = std::chrono::high_resolution_clock::now();

            // do render
            testStuff(renderer);
        }
    }
    catch (std::runtime_error e)
    {
        std::cout << e.what() << std::endl;
    }
    glfwTerminate();
    return 0;
}
