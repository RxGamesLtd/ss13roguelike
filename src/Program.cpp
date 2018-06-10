#include "config.hpp"
#include "renderer.hpp"
#include "material.hpp"

#define GLFW_INCLUDE_VULKAN
#include "GLFW/glfw3.h"

#include "targetver.hpp"
#include <iostream>
#include <thread>

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

int main(int argc, char *argv[])
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
        const auto window = glfwCreateWindow(1024, 768, "TestApp:Initilizing", nullptr, nullptr);

        glfwSetKeyCallback(window, keyCallback);

        const auto instanceExtensions = getRequiredInstanceExtensions();
        const auto deviceExtensions   = getRequiredDeviceExtensions();

        auto name = std::string("SS13Roguelike");
        // init renderer
        auto renderer = Renderer(name, window, instanceExtensions, deviceExtensions);

        // init done
        glfwSetWindowTitle(window, name.c_str());

        const auto mat = Material(renderer, "triangle");
        renderer.prepairFor(mat);

        const auto targetRenderTime = static_cast<int64_t>(0.016666 * 1000000);
        while (!glfwWindowShouldClose(window))
        {
            const auto frameStart = std::chrono::steady_clock::now();
            glfwPollEvents();

            // do render
            renderer.beginRender();

            renderer.draw(3, 1, 0, 0);

            renderer.endRender();
            renderer.present();

            const auto renderTime = std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::steady_clock::now() - frameStart).count();
            if(targetRenderTime > renderTime)
            {
                std::this_thread::sleep_for(std::chrono::microseconds(targetRenderTime - renderTime));
            }
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
