#include "config.hpp"
#include "renderer.hpp"
#include "material.hpp"

#include "targetver.hpp"

#define GLFW_INCLUDE_VULKAN
#include "GLFW/glfw3.h"
#include "glm/glm.hpp"
#include "vulkan/vulkan.hpp"

#include <iostream>

static void testStuff(Renderer& r)
{
    r.beginRender();

    r.m_commandBuffers[r.m_currentImageIndex]->draw(3, 1, 0, 0);

    r.endRender();
    r.present();
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
    std::ios_base::sync_with_stdio(false);
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


        auto mat = Material(renderer, "triangle");
        renderer.prepairFor(mat);

        while (!glfwWindowShouldClose(window))
        {
            glfwPollEvents();
            renderer.m_frameStartTime = std::chrono::high_resolution_clock::now();

            // do render
            testStuff(renderer);

            auto renderTime = std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::high_resolution_clock::now() - renderer.m_frameStartTime).count();
            double renderTimeD = renderTime / 1000000.0;
            std::cout << "TPF " << renderTimeD << " FPS " << (1 / renderTimeD) << "\n";
        }
        renderer.waitForIdle();
    }
    catch (std::runtime_error e)
    {
        std::cout << e.what() << std::endl;
    }
    glfwTerminate();
    return 0;
}
