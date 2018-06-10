#include "renderer.hpp"

#include "config.hpp"
#include "material.hpp"

#define GLFW_INCLUDE_VULKAN
#include "GLFW/glfw3.h"
#include "vulkan/vulkan.hpp"

#include <iostream>
#include <set>

namespace
{
vk::Extent2D getWindowSize(GLFWwindow* window)
{
    int width;
    int height;

    glfwGetWindowSize(window, &width, &height);
    return { static_cast<uint32_t>(width), static_cast<uint32_t>(height) };
}

vk::SurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<vk::SurfaceFormatKHR>& availableFormats)
{
    if (availableFormats.size() == 1 && availableFormats[0].format == vk::Format::eUndefined)
    {
        return { vk::Format::eB8G8R8A8Unorm, vk::ColorSpaceKHR::eSrgbNonlinear };
    }

    for (const auto& availableFormat : availableFormats)
    {
        if (availableFormat.format == vk::Format::eB8G8R8A8Unorm &&
            availableFormat.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear)
        {
            return availableFormat;
        }
    }

    return availableFormats[0];
}

vk::PresentModeKHR chooseSwapPresentMode(const std::vector<vk::PresentModeKHR> availablePresentModes)
{
    auto bestMode = vk::PresentModeKHR::eFifo;

    for (const auto& availablePresentMode : availablePresentModes)
    {
        if (availablePresentMode == vk::PresentModeKHR::eMailbox)
        {
            return availablePresentMode;
        }
        if (availablePresentMode == vk::PresentModeKHR::eImmediate)
        {
            bestMode = availablePresentMode;
        }
    }

    return bestMode;
}

vk::Extent2D chooseSwapExtent(vk::Extent2D desiredExtent, const vk::SurfaceCapabilitiesKHR& capabilities)
{
    if (capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max())
    {
        return capabilities.currentExtent;
    }
    desiredExtent.width = std::clamp(desiredExtent.width, capabilities.minImageExtent.width, capabilities.maxImageExtent.width);
    desiredExtent.height = std::clamp(desiredExtent.height, capabilities.minImageExtent.height, capabilities.maxImageExtent.height);

    return desiredExtent;
}
}

Renderer::Renderer(
    const std::string& appName, GLFWwindow* window, const std::vector<const char*>& instanceExtensions,
    const std::vector<const char*>& requestedDeviceExtensions)
{
    try
    {
        initInstance(appName, instanceExtensions);
        initSurface(window);
        initDevice(requestedDeviceExtensions);
        initSwapchain(getWindowSize(window));
        initImageViews();
        initCommandPool();

        m_isInited = true;
    }
    catch (std::exception e)
    {
        m_isInited = false;
    }
}

void Renderer::initInstance(const std::string& appName, const std::vector<const char*>& instanceExtensions)
{
    // Use validation layers if this is a debug build
    auto layers = std::vector<const char*>();

    if constexpr (Config::isDebug)
    {
        const auto instanceLayerProps = vk::enumerateInstanceLayerProperties();
        for (const auto& lp : instanceLayerProps)
        {
            const std::string lpName(lp.layerName);

            std::cout << "Instance Layer: " << lpName << "\t|\t" << lp.description << "\n";

            if (lpName.find("VK_LAYER_LUNARG_standard_validation") != std::string::npos)
            {
                layers.push_back("VK_LAYER_LUNARG_standard_validation");
            }
            if (lpName.find("VK_LAYER_LUNARG_core_validation") != std::string::npos)
            {
                layers.push_back("VK_LAYER_LUNARG_core_validation");
            }
        }
        std::cout << std::endl;
    }

    // VkApplicationInfo allows the programmer to specifiy some basic information
    // about the program, which can be useful for layers and tools to provide
    // more debug information.
    const auto appInfo = vk::ApplicationInfo()
        .setPApplicationName(appName.c_str())
        .setApplicationVersion(VK_MAKE_VERSION(1, 0, 0))
        .setPEngineName("LunarG SDK")
        .setEngineVersion(VK_MAKE_VERSION(1, 0, 0))
        .setApiVersion(VK_API_VERSION_1_0);

    // VkInstanceCreateInfo is where the programmer specifies the layers and/or
    // extensions that are needed. For now, none are enabled.
    const auto instInfo = vk::InstanceCreateInfo()
        .setFlags({})
        .setPApplicationInfo(&appInfo)
        .setEnabledExtensionCount(static_cast<uint32_t>(instanceExtensions.size()))
        .setPpEnabledExtensionNames(instanceExtensions.data())
        .setEnabledLayerCount(static_cast<uint32_t>(layers.size()))
        .setPpEnabledLayerNames(layers.data());

    // Create the Vulkan instance.
    try
    {
        m_instance = vk::createInstanceUnique(instInfo);
    }
    catch (std::exception e)
    {
        std::cout << "Could not create a Vulkan instance: " << e.what() << std::endl;
        throw;
    }
}

void Renderer::initSurface(GLFWwindow* window)
{
    VkSurfaceKHR surf{};

    if (glfwCreateWindowSurface(VkInstance(m_instance.get()), window, nullptr, &surf) != VK_SUCCESS)
    {
        std::cout << "Error creating surface" << std::endl;
        throw;
    }
    m_surface = vk::UniqueSurfaceKHR(vk::SurfaceKHR(surf), { m_instance.get() });
}

bool Renderer::fillQueueFamilies(vk::PhysicalDevice& gpu)
{
    auto graphicsFamilyIdx = UINT32_MAX;
    auto presentFamilyIdx = UINT32_MAX;
    const auto queues = gpu.getQueueFamilyProperties();
    {
        const auto pred = [&](const auto& d) {
                return d.queueFlags & vk::QueueFlagBits::eGraphics;
            };

        const auto q = std::find_if(
            queues.begin(), queues.end(), pred);
        if (q != queues.end())
        {
            graphicsFamilyIdx = static_cast<uint32_t>(q - queues.begin());
        }
    }
    {
        const auto pred = [&](const auto& d) {
                const auto curIndex = static_cast<uint32_t>(&d - &queues[0]);
                const auto presentSupport =
                    glfwGetPhysicalDevicePresentationSupport(VkInstance(m_instance.get()), VkPhysicalDevice(gpu), curIndex) == GLFW_TRUE;
                const auto surfaceSupport = !m_surface || gpu.getSurfaceSupportKHR(curIndex, m_surface.get());

                return surfaceSupport && presentSupport;
            };

        const auto q = std::find_if(queues.begin(), queues.end(), pred);
        if (q != queues.end())
        {
            presentFamilyIdx = static_cast<uint32_t>(q - queues.begin());
        }
    }

    if (graphicsFamilyIdx != UINT32_MAX && presentFamilyIdx != UINT32_MAX)
    {
        m_graphicsFamilyIdx = graphicsFamilyIdx;
        m_presentFamilyIdx = presentFamilyIdx;
        return true;
    }
    return false;
}

bool Renderer::isDeviceCompatible(const vk::PhysicalDevice& device, const std::vector<const char*>& requestedDeviceExtensions)
{
    // check extensions
    const auto extensions = device.enumerateDeviceExtensionProperties();
    auto hasExtensions = true;

    for (const auto& reqExt : requestedDeviceExtensions)
    {
        if (std::find_if(extensions.begin(), extensions.end(), [&reqExt](vk::ExtensionProperties vl) {
            return strcmp(vl.extensionName, reqExt) == 0;
        }) == extensions.end())
        {
            hasExtensions = false;
            break;
        }
    }
    // check discreteness
    const auto props = device.getProperties();
    const bool isDeviceDiscrete = props.deviceType == vk::PhysicalDeviceType::eDiscreteGpu;

    // check swapchain support
    bool isSwapchainNormal = false;
    if (hasExtensions)
    {
        auto formats = device.getSurfaceFormatsKHR(m_surface.get());
        auto presentModes = device.getSurfacePresentModesKHR(m_surface.get());

        isSwapchainNormal = !formats.empty() && !presentModes.empty();
    }
    return isDeviceDiscrete && hasExtensions && isSwapchainNormal;
}

void Renderer::initDevice(const std::vector<const char*>& requestedDeviceExtensions)
{
    auto gpus = m_instance->enumeratePhysicalDevices();

    // Find physical device
    for (auto& device : gpus)
    {
        const auto props = device.getProperties();

        std::cout << "Device: " << props.deviceName << "\n";
        std::cout << "\tType:" << static_cast<uint32_t>(props.deviceType) << "\n";
        std::cout << "\tDriver:" << props.driverVersion << "\n";

        const bool isCompatible = isDeviceCompatible(device, requestedDeviceExtensions);
        std::cout << "\tCompatible:" << (isCompatible ? "true" : "false") << "\n";

        if (isCompatible && fillQueueFamilies(device))
        {
            m_gpu = device;
        }
    }
    if (!m_gpu)
    {
        std::cout << "Could not find an appropriate physical device!" << std::endl;
        throw;
    }

    auto queueCreateInfos = std::vector<vk::DeviceQueueCreateInfo>();
    auto uniqueQueueFamilies = std::set<uint32_t>{ m_presentFamilyIdx, m_graphicsFamilyIdx };

    auto queuePriority = 1.0f;
    for (const auto& queueFamily : uniqueQueueFamilies)
    {
        const auto devQueueCreateInfo = vk::DeviceQueueCreateInfo()
            .setQueueFamilyIndex(queueFamily)
            .setQueueCount(1)
            .setPQueuePriorities(&queuePriority);
        queueCreateInfos.push_back(devQueueCreateInfo);
    }

    const auto devCreateInfo = vk::DeviceCreateInfo()
        .setQueueCreateInfoCount(static_cast<uint32_t>(queueCreateInfos.size()))
        .setPQueueCreateInfos(queueCreateInfos.data())
        .setEnabledExtensionCount(static_cast<uint32_t>(requestedDeviceExtensions.size()))
        .setPpEnabledExtensionNames(requestedDeviceExtensions.data());

    try
    {
        m_device = m_gpu.createDeviceUnique(devCreateInfo);
    }
    catch (std::exception e)
    {
        std::cout << "Could not create a Vulkan device: " << e.what() << std::endl;
        throw;
    }

    try
    {
        m_queue = m_device->getQueue(m_graphicsFamilyIdx, 0u);
        m_presentQueue = m_device->getQueue(m_presentFamilyIdx, 0u);
    }
    catch (std::exception e)
    {
        std::cout << "Could not get a Vulkan queue: " << e.what() << std::endl;
        throw;
    }
}

void Renderer::initSwapchain(vk::Extent2D desiredExtent)
{
    const auto formats = m_gpu.getSurfaceFormatsKHR(m_surface.get());
    const auto presentModes = m_gpu.getSurfacePresentModesKHR(m_surface.get());
    const auto capabilities = m_gpu.getSurfaceCapabilitiesKHR(m_surface.get());

    const auto surfaceFormat = chooseSwapSurfaceFormat(formats);
    const auto presentMode = chooseSwapPresentMode(presentModes);
    const auto extent = chooseSwapExtent(desiredExtent, capabilities);

    auto imageCount = capabilities.minImageCount + 1;

    if (capabilities.maxImageCount > 0 && imageCount > capabilities.maxImageCount)
    {
        imageCount = capabilities.maxImageCount;
    }

    auto swapCreateInfo = vk::SwapchainCreateInfoKHR()
        .setMinImageCount(imageCount)
        .setImageFormat(surfaceFormat.format)
        .setImageColorSpace(surfaceFormat.colorSpace)
        .setImageExtent(extent)
        .setImageArrayLayers(1)
        .setImageUsage(vk::ImageUsageFlagBits::eColorAttachment)
        .setPreTransform(capabilities.currentTransform)
        .setCompositeAlpha(vk::CompositeAlphaFlagBitsKHR::eOpaque)
        .setPresentMode(presentMode)
        .setClipped(true)
        .setOldSwapchain(m_swapchain.get())
        .setSurface(m_surface.get());

    auto queueFamilyIndices = std::array<uint32_t, 2>{ m_graphicsFamilyIdx, m_presentFamilyIdx };
    if (m_graphicsFamilyIdx != m_presentFamilyIdx)
    {
        swapCreateInfo
        .setImageSharingMode(vk::SharingMode::eConcurrent)
        .setQueueFamilyIndexCount(static_cast<uint32_t>(queueFamilyIndices.size()))
        .setPQueueFamilyIndices(queueFamilyIndices.data());
    }
    else
    {
        swapCreateInfo
        .setImageSharingMode(vk::SharingMode::eExclusive)
        .setQueueFamilyIndexCount(0)
        .setPQueueFamilyIndices(nullptr);
    }

    try
    {
        m_swapchain = m_device->createSwapchainKHRUnique(swapCreateInfo);
    }
    catch (std::exception e)
    {
        std::cout << "Could not create a Vulkan swapchain: " << e.what() << std::endl;
        throw;
    }

    try
    {
        m_swapchainImages = m_device->getSwapchainImagesKHR(m_swapchain.get());
    }
    catch (std::exception e)
    {
        std::cout << "Could not get a Vulkan swapchain images: " << e.what() << std::endl;
        throw;
    }

    m_swapchainFormat = surfaceFormat.format;
    m_swapchainExtent = extent;
}

void Renderer::initImageViews()
{
    m_swapChainImageViews.reserve(m_swapchainImages.size());

    for (const auto& image : m_swapchainImages)
    {
        const auto imageViewCreateInfo = vk::ImageViewCreateInfo()
            .setViewType(vk::ImageViewType::e2D)
            .setFormat(m_swapchainFormat)
            .setComponents(
            vk::ComponentMapping(
                vk::ComponentSwizzle::eIdentity,
                vk::ComponentSwizzle::eIdentity,
                vk::ComponentSwizzle::eIdentity,
                vk::ComponentSwizzle::eIdentity))
            .setSubresourceRange(
            vk::ImageSubresourceRange()
            .setAspectMask(vk::ImageAspectFlagBits::eColor)
            .setBaseArrayLayer(0)
            .setBaseMipLevel(0)
            .setLayerCount(1)
            .setLevelCount(1))
            .setImage(image);

        try
        {
            m_swapChainImageViews.push_back(m_device->createImageViewUnique(imageViewCreateInfo));
        }
        catch (std::exception e)
        {
            std::cout << "Could not get a Vulkan swapchain image view: " << e.what() << std::endl;
            throw;
        }
    }
}

void Renderer::prepairFor(const Material& mat)
{
    initRenderPass();
    initPipeline(mat);
    initFramebuffers();
    initCommandBuffers();
    initSyncObjects();
}

void Renderer::beginRender()
{
    m_device->waitForFences(m_inFlightFences[m_currentFrameIndex].get(), true, UINT64_MAX);
    m_device->resetFences(m_inFlightFences[m_currentFrameIndex].get());

    const auto imageIndex = m_device->acquireNextImageKHR(
        m_swapchain.get(), UINT64_MAX, m_imageAvailableSemaphores[m_currentFrameIndex].get(), nullptr);

    if (imageIndex.result != vk::Result::eSuccess)
    {
        throw std::exception("Failed to prepair rendering");
    }

    m_currentImageIndex = imageIndex.value;

    const auto cbbi = vk::CommandBufferBeginInfo()
        .setFlags(vk::CommandBufferUsageFlagBits::eSimultaneousUse);

    m_commandBuffers[m_currentImageIndex]->begin(cbbi);

    const auto clearValue = vk::ClearValue(std::array<float, 4>{ 0.f, 0.f, 0.f, 0.f });
    const auto rpbi = vk::RenderPassBeginInfo()
        .setRenderPass(m_renderPass.get())
        .setFramebuffer(m_framebuffers[m_currentImageIndex].get())
        .setRenderArea({{ 0, 0 }, m_swapchainExtent })
        .setClearValueCount(1)
        .setPClearValues(&clearValue);

    m_commandBuffers[m_currentImageIndex]->beginRenderPass(rpbi, vk::SubpassContents::eInline);

    m_commandBuffers[m_currentImageIndex]->bindPipeline(vk::PipelineBindPoint::eGraphics, m_graphicsPipeline.get());
}

void Renderer::draw(uint32_t vertexCount, uint32_t instanceCount, uint32_t firstVertex, uint32_t firstInstance)
{
    m_commandBuffers[m_currentImageIndex]->draw(vertexCount, instanceCount, firstVertex, firstInstance);
}

void Renderer::endRender()
{
    m_commandBuffers[m_currentImageIndex]->endRenderPass();
    m_commandBuffers[m_currentImageIndex]->end();

    auto submitCommandBuffer = std::vector<vk::CommandBuffer>{ m_commandBuffers[m_currentImageIndex].get() };
    auto waitStages = std::array<vk::PipelineStageFlags, 1>{ vk::PipelineStageFlagBits::eColorAttachmentOutput };
    auto waitSemaphores = std::array<vk::Semaphore, 1>{ m_imageAvailableSemaphores[m_currentFrameIndex].get() };
    auto signalSemaphores = std::array<vk::Semaphore, 1> { m_renderFinishedSemaphores[m_currentFrameIndex].get() };

    const auto submitInfo = vk::SubmitInfo()
        .setPCommandBuffers(submitCommandBuffer.data())
        .setCommandBufferCount(static_cast<uint32_t>(submitCommandBuffer.size()))
        .setPWaitSemaphores(waitSemaphores.data())
        .setWaitSemaphoreCount(static_cast<uint32_t>(waitSemaphores.size()))
        .setPWaitDstStageMask(waitStages.data())
        .setPSignalSemaphores(signalSemaphores.data())
        .setSignalSemaphoreCount(static_cast<uint32_t>(signalSemaphores.size()));

    m_queue.submit(submitInfo, m_inFlightFences[m_currentImageIndex].get());
}

void Renderer::present()
{
    const auto presentInfo = vk::PresentInfoKHR()
        .setSwapchainCount(1)
        .setPSwapchains(&m_swapchain.get())
        .setPImageIndices(&m_currentImageIndex)
        .setPWaitSemaphores(&m_renderFinishedSemaphores[m_currentFrameIndex].get())
        .setWaitSemaphoreCount(1);

    if (m_presentQueue.presentKHR(presentInfo) != vk::Result::eSuccess)
    {
        throw std::runtime_error("Cannot present an image.");
    }
    m_currentFrameIndex = (m_currentFrameIndex + 1) % m_inFlightFences.size();
}

void Renderer::waitForIdle()
{
    m_device->waitIdle();
}

void Renderer::initRenderPass()
{
    const auto colorAttachment = vk::AttachmentDescription()
        .setFormat(m_swapchainFormat)
        .setSamples(vk::SampleCountFlagBits::e1)
        .setLoadOp(vk::AttachmentLoadOp::eClear)
        .setStoreOp(vk::AttachmentStoreOp::eStore)
        .setStencilLoadOp(vk::AttachmentLoadOp::eDontCare)
        .setStencilStoreOp(vk::AttachmentStoreOp::eDontCare)
        .setInitialLayout(vk::ImageLayout::eUndefined)
        .setFinalLayout(vk::ImageLayout::ePresentSrcKHR);

    const auto colorAttachmentRef = vk::AttachmentReference()
        .setAttachment(0)
        .setLayout(vk::ImageLayout::eColorAttachmentOptimal);

    const auto dependency = vk::SubpassDependency()
        .setSrcSubpass(VK_SUBPASS_EXTERNAL)
        .setSrcStageMask(vk::PipelineStageFlagBits::eColorAttachmentOutput)
        .setSrcAccessMask({})
        .setDstSubpass(0)
        .setDstStageMask(vk::PipelineStageFlagBits::eColorAttachmentOutput)
        .setDstAccessMask(vk::AccessFlagBits::eColorAttachmentRead | vk::AccessFlagBits::eColorAttachmentWrite);

    const auto colorAttachments = std::array<vk::AttachmentDescription, 1>{ colorAttachment };
    const auto colorAttachmentRefs = std::array<vk::AttachmentReference, 1>{ colorAttachmentRef };
    const auto dependencies = std::array<vk::SubpassDependency, 1>{ dependency };

    const auto subpass = vk::SubpassDescription()
        .setPipelineBindPoint(vk::PipelineBindPoint::eGraphics)
        .setColorAttachmentCount(static_cast<uint32_t>(colorAttachmentRefs.size()))
        .setPColorAttachments(colorAttachmentRefs.data());

    const auto subpasses = std::array<vk::SubpassDescription, 1>{ subpass };

    const auto renderPassCreateInfo = vk::RenderPassCreateInfo()
        .setAttachmentCount(static_cast<uint32_t>(colorAttachments.size()))
        .setPAttachments(colorAttachments.data())
        .setSubpassCount(static_cast<uint32_t>(subpasses.size()))
        .setPSubpasses(subpasses.data())
        .setDependencyCount(static_cast<uint32_t>(dependencies.size()))
        .setPDependencies(dependencies.data());

    try
    {
        m_renderPass = m_device->createRenderPassUnique(renderPassCreateInfo);
    }
    catch (std::exception e)
    {
        std::cout << "Could not create a Vulkan render pass: " << e.what() << std::endl;
        throw;
    }
}

void Renderer::initPipelineCache()
{
    const auto pipelineCacheCreateInfo = vk::PipelineCacheCreateInfo()
        .setInitialDataSize(0U)
        .setPInitialData(nullptr);

    try
    {
        m_pipelineCache = m_device->createPipelineCacheUnique(pipelineCacheCreateInfo);
    }
    catch (std::exception e)
    {
        std::cout << "Could not create a Vulkan pipelines cache: " << e.what() << std::endl;
        throw;
    }
}

void Renderer::initPipeline(const Material& mat)
{
    const auto[vertShader, fragShader] = mat.getShaders();

    const auto vertShaderStageInfo = vk::PipelineShaderStageCreateInfo()
        .setStage(vk::ShaderStageFlagBits::eVertex)
        .setModule(vertShader)
        .setPName("main");

    const auto fragShaderBinStageInfo = vk::PipelineShaderStageCreateInfo()
        .setStage(vk::ShaderStageFlagBits::eFragment)
        .setModule(fragShader)
        .setPName("main");

    const auto shaderStages = std::array<vk::PipelineShaderStageCreateInfo, 2>{ vertShaderStageInfo, fragShaderBinStageInfo };

    const auto vertexInputInfo = vk::PipelineVertexInputStateCreateInfo()
        .setVertexBindingDescriptionCount(0)
        .setVertexAttributeDescriptionCount(0);

    const auto inputAssembly = vk::PipelineInputAssemblyStateCreateInfo()
        .setTopology(vk::PrimitiveTopology::eTriangleList)
        .setPrimitiveRestartEnable(false);

    const auto width = static_cast<float>(m_swapchainExtent.width);
    const auto height = static_cast<float>(m_swapchainExtent.height);

    const auto viewports = std::array<vk::Viewport, 1>{ vk::Viewport{ 0.0f, 0.0f, width, height, 0.0f, 1.0f } };
    const auto scissors = std::array<vk::Rect2D, 1>{ vk::Rect2D{{ 0, 0 }, m_swapchainExtent } };

    const auto viewportState = vk::PipelineViewportStateCreateInfo()
        .setViewportCount(static_cast<uint32_t>(viewports.size()))
        .setPViewports(viewports.data())
        .setScissorCount(static_cast<uint32_t>(scissors.size()))
        .setPScissors(scissors.data());

    const auto rasterizer = vk::PipelineRasterizationStateCreateInfo()
        .setDepthClampEnable(false)
        .setRasterizerDiscardEnable(false)
        .setPolygonMode(vk::PolygonMode::eFill)
        .setLineWidth(1.0f)
        .setCullMode(vk::CullModeFlagBits::eBack)
        .setFrontFace(vk::FrontFace::eClockwise)
        .setDepthBiasEnable(true);

    const auto multisampling = vk::PipelineMultisampleStateCreateInfo()
        .setSampleShadingEnable(false)
        .setRasterizationSamples(vk::SampleCountFlagBits::e1);

    const auto colorBlendAttachment = vk::PipelineColorBlendAttachmentState()
        .setColorWriteMask(
        vk::ColorComponentFlagBits::eR |
        vk::ColorComponentFlagBits::eG |
        vk::ColorComponentFlagBits::eB |
        vk::ColorComponentFlagBits::eA)
        .setBlendEnable(false);

    const auto blendAttachments = std::array<vk::PipelineColorBlendAttachmentState, 1>{ colorBlendAttachment };

    const auto colorBlending = vk::PipelineColorBlendStateCreateInfo()
        .setLogicOpEnable(false)
        .setLogicOp(vk::LogicOp::eCopy)
        .setAttachmentCount(static_cast<uint32_t>(blendAttachments.size()))
        .setPAttachments(blendAttachments.data())
        .setBlendConstants({ 0.0f, 0.0f, 0.0f, 0.0f });

    const auto pipelineLayoutCreateInfo = vk::PipelineLayoutCreateInfo()
        .setSetLayoutCount(0U)
        .setPushConstantRangeCount(0U);

    try
    {
        m_pipelineLayout = m_device->createPipelineLayoutUnique(pipelineLayoutCreateInfo);
    }
    catch (std::exception e)
    {
        std::cout << "Could not create a Vulkan pipeline layout: " << e.what() << std::endl;
        throw;
    }

    const auto pipelineCreateInfo = vk::GraphicsPipelineCreateInfo()
        .setStageCount(static_cast<uint32_t>(shaderStages.size()))
        .setPStages(shaderStages.data())
        .setPVertexInputState(&vertexInputInfo)
        .setPInputAssemblyState(&inputAssembly)
        .setPViewportState(&viewportState)
        .setPRasterizationState(&rasterizer)
        .setPMultisampleState(&multisampling)
        .setPColorBlendState(&colorBlending)
        .setLayout(m_pipelineLayout.get())
        .setRenderPass(m_renderPass.get())
        .setSubpass(0)
        .setBasePipelineHandle(nullptr);

    try
    {
        m_graphicsPipeline = m_device->createGraphicsPipelineUnique(m_pipelineCache.get(), pipelineCreateInfo);
    }
    catch (std::exception e)
    {
        std::cout << "Could not create a Vulkan graphics pipeline: " << e.what() << std::endl;
        throw;
    }
}

void Renderer::initFramebuffers()
{
    m_framebuffers.resize(m_swapChainImageViews.size());
    for (size_t i = 0; i < m_swapChainImageViews.size(); i++)
    {
        const auto attachments = std::array<vk::ImageView, 1>{ m_swapChainImageViews[i].get() };

        const auto framebufferCreateInfo = vk::FramebufferCreateInfo()
            .setRenderPass(m_renderPass.get())
            .setAttachmentCount(static_cast<uint32_t>(attachments.size()))
            .setPAttachments(attachments.data())
            .setWidth(m_swapchainExtent.width)
            .setHeight(m_swapchainExtent.height)
            .setLayers(1);

        try
        {
            m_framebuffers[i] = m_device->createFramebufferUnique(framebufferCreateInfo);
        }
        catch (std::exception e)
        {
            std::cout << "Could not create a Vulkan framebuffer: " << e.what() << std::endl;
            throw;
        }
    }
}

void Renderer::initCommandPool()
{
    const auto cpci = vk::CommandPoolCreateInfo()
        .setQueueFamilyIndex(m_graphicsFamilyIdx)
        .setFlags(vk::CommandPoolCreateFlagBits::eResetCommandBuffer);

    m_commandPool = m_device->createCommandPoolUnique(cpci);
}

void Renderer::initCommandBuffers()
{
    const auto cbci = vk::CommandBufferAllocateInfo()
        .setCommandPool(m_commandPool.get())
        .setCommandBufferCount(static_cast<uint32_t>(m_framebuffers.size()))
        .setLevel(vk::CommandBufferLevel::ePrimary);

    m_commandBuffers = m_device->allocateCommandBuffersUnique(cbci);
}

void Renderer::initSyncObjects()
{
    const auto sci = vk::SemaphoreCreateInfo();
    const auto fci = vk::FenceCreateInfo(vk::FenceCreateFlagBits::eSignaled);

    for (size_t i = 0; i < m_commandBuffers.size(); ++i)
    {
        m_imageAvailableSemaphores.push_back(m_device->createSemaphoreUnique(sci));
        m_renderFinishedSemaphores.push_back(m_device->createSemaphoreUnique(sci));
        m_inFlightFences.push_back(m_device->createFenceUnique(fci));
    }
}
