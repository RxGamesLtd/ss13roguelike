#include "renderer.hpp"

#include "config.hpp"
#include "material.hpp"

#define GLFW_INCLUDE_VULKAN
#include "GLFW/glfw3.h"
#include "vulkan/vulkan.hpp"

#include <assert.h>
#include <iostream>
#include <set>

vk::Extent2D getWindowSize(GLFWwindow* window)
{
    int width;
    int height;
    glfwGetWindowSize(window, &width, &height);
    return { static_cast<uint32_t>(width), static_cast<uint32_t>(height) };
}

Renderer::Renderer(GLFWwindow* window, const std::vector<const char*>& instanceExtensions,
                   const std::vector<const char*>& requestedDeviceExtensions)
{
    try
    {
        initInstance(instanceExtensions);
        initSurface(window);
        initDevice(requestedDeviceExtensions);
        initSwapchain(getWindowSize(window));
        initImageViews();
        initCommandPool();
        initSyncObjects();

        m_isInited = true;
    }
    catch(std::exception e)
    {
        m_isInited = false;
    }
}

void Renderer::initInstance(const std::vector<const char*>& instanceExtensions)
{
    // Use validation layers if this is a debug build
    std::vector<const char*> layers;
    if constexpr (Config::isDebug)
    {
        const auto instanceLayerProps = vk::enumerateInstanceLayerProperties();
        for(const auto& lp : instanceLayerProps)
        {
            const std::string lpName(lp.layerName);

            std::cout << "Instance Layer: " << lpName << "\t|\t" << lp.description << "\n";

            if(lpName.find("VK_LAYER_LUNARG_standard_validation") != std::string::npos)
            {
                layers.push_back("VK_LAYER_LUNARG_standard_validation");
            }
            if(lpName.find("VK_LAYER_LUNARG_core_validation") != std::string::npos)
            {
                layers.push_back("VK_LAYER_LUNARG_core_validation");
            }
        }
        std::cout << std::endl;
    }

    // VkApplicationInfo allows the programmer to specifiy some basic information
    // about the program, which can be useful for layers and tools to provide
    // more debug information.
    auto appInfo = vk::ApplicationInfo();
    appInfo.setPApplicationName("SS13 rogue-like");
    appInfo.setApplicationVersion(VK_MAKE_VERSION(1, 0, 0));
    appInfo.setPEngineName("LunarG SDK");
    appInfo.setEngineVersion(1);
    appInfo.setApiVersion(VK_API_VERSION_1_0);

    // VkInstanceCreateInfo is where the programmer specifies the layers and/or
    // extensions that are needed. For now, none are enabled.
    auto instInfo = vk::InstanceCreateInfo();
    instInfo.setFlags(vk::InstanceCreateFlags());
    instInfo.setPApplicationInfo(&appInfo);
    instInfo.setEnabledExtensionCount(static_cast<uint32_t>(instanceExtensions.size()));
    instInfo.setPpEnabledExtensionNames(instanceExtensions.data());
    instInfo.setEnabledLayerCount(static_cast<uint32_t>(layers.size()));
    instInfo.setPpEnabledLayerNames(layers.data());

    // Create the Vulkan instance.
    try
    {
        m_instance = vk::createInstanceUnique(instInfo);
    }
    catch(std::exception e)
    {
        std::cout << "Could not create a Vulkan instance: " << e.what() << std::endl;
        throw;
    }
}

void Renderer::initSurface(GLFWwindow* window)
{
    VkSurfaceKHR surf{};
    if(glfwCreateWindowSurface(VkInstance(m_instance.get()), window, nullptr, &surf) != VK_SUCCESS)
    {
        std::cout << "Error creating surface" << std::endl;
        throw;
    }
    //vk::SurfaceKHRDeleter deleter(m_instance.get(), nullptr);
    m_surface = vk::UniqueSurfaceKHR(vk::SurfaceKHR(surf), m_instance.get());
}

bool Renderer::fillQueueFamilies(vk::PhysicalDevice& gpu)
{
    uint32_t graphicsFamilyIdx = UINT32_MAX;
    uint32_t presentFamilyIdx = UINT32_MAX;
    const auto queues = gpu.getQueueFamilyProperties();
    {
        const auto q = std::find_if(
          queues.begin(), queues.end(), [&](const auto& d) { return d.queueFlags & vk::QueueFlagBits::eGraphics; });

        if(q != queues.end())
        {
            graphicsFamilyIdx = static_cast<uint32_t>(q - queues.begin());
        }
    }
    {
        const auto q = std::find_if(queues.begin(), queues.end(), [&](const auto& d) {
            const auto curIndex = static_cast<uint32_t>(&d - &queues[0]);
            bool presentSupport =
              glfwGetPhysicalDevicePresentationSupport(VkInstance(m_instance.get()), VkPhysicalDevice(gpu), curIndex);
            bool surfaceSupport = !m_surface || gpu.getSurfaceSupportKHR(curIndex, m_surface.get());
            return surfaceSupport && presentSupport;
        });

        if(q != queues.end())
        {
            presentFamilyIdx = static_cast<uint32_t>(q - queues.begin());
        }
    }
    if(graphicsFamilyIdx != UINT32_MAX && presentFamilyIdx != UINT32_MAX)
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
    bool hasExtensions = true;
    for(const auto& reqExt : requestedDeviceExtensions)
    {
        if(std::find_if(extensions.begin(), extensions.end(), [&reqExt](vk::ExtensionProperties vl) {
               return strcmp(vl.extensionName, reqExt) == 0;
           }) == extensions.end())
        {
            hasExtensions = false;
            break;
        }
    }
    // check discreteness
    const auto props = device.getProperties();
    bool isDeviceDiscrete = props.deviceType == vk::PhysicalDeviceType::eDiscreteGpu;

    // check swapchain support
    bool isSwapchainNormal = false;
    if(hasExtensions)
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
    for(auto& device : gpus)
    {
        const auto props = device.getProperties();

        std::cout << "Device: " << props.deviceName << "\n";
        std::cout << "\tType:" << static_cast<uint32_t>(props.deviceType) << "\n";
        std::cout << "\tDriver:" << props.driverVersion << "\n";

        bool isCompatible = isDeviceCompatible(device, requestedDeviceExtensions);
        std::cout << "\tCompatible:" << (isCompatible ? "true" : "false") << "\n";

        if(isCompatible && fillQueueFamilies(device))
        {
            m_gpu = device;
        }
    }
    if(!m_gpu)
    {
        std::cout << "Could not find an appropriate physical device!" << std::endl;
        throw;
    }

    std::vector<vk::DeviceQueueCreateInfo> queueCreateInfos;
    std::set<uint32_t> uniqueQueueFamilies = { m_presentFamilyIdx, m_graphicsFamilyIdx };

    float queuePriority = 1.0f;
    for(const auto& queueFamily : uniqueQueueFamilies)
    {
        auto devQueueCreateInfo = vk::DeviceQueueCreateInfo();
        devQueueCreateInfo.setQueueFamilyIndex(queueFamily);
        devQueueCreateInfo.setQueueCount(1);
        devQueueCreateInfo.setPQueuePriorities(&queuePriority);
        queueCreateInfos.push_back(devQueueCreateInfo);
    }

    auto devCreateInfo = vk::DeviceCreateInfo();
    devCreateInfo.setQueueCreateInfoCount(static_cast<uint32_t>(queueCreateInfos.size()));
    devCreateInfo.setPQueueCreateInfos(queueCreateInfos.data());
    devCreateInfo.setEnabledExtensionCount(static_cast<uint32_t>(requestedDeviceExtensions.size()));
    devCreateInfo.setPpEnabledExtensionNames(requestedDeviceExtensions.data());

    try
    {
        m_device = m_gpu.createDeviceUnique(devCreateInfo);
    }
    catch(std::exception e)
    {
        std::cout << "Could not create a Vulkan device: " << e.what() << std::endl;
        throw;
    }

    try
    {
        m_queue = m_device->getQueue(m_graphicsFamilyIdx, 0u);
        m_presentQueue = m_device->getQueue(m_presentFamilyIdx, 0u);
    }
    catch(std::exception e)
    {
        std::cout << "Could not get a Vulkan queue: " << e.what() << std::endl;
        throw;
    }
}

vk::SurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<vk::SurfaceFormatKHR>& availableFormats)
{
    if(availableFormats.size() == 1 && availableFormats[0].format == vk::Format::eUndefined)
    {
        return { vk::Format::eB8G8R8A8Unorm, vk::ColorSpaceKHR::eSrgbNonlinear };
    }

    for(const auto& availableFormat : availableFormats)
    {
        if(availableFormat.format == vk::Format::eB8G8R8A8Unorm &&
           availableFormat.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear)
        {
            return availableFormat;
        }
    }

    return availableFormats[0];
}

vk::PresentModeKHR chooseSwapPresentMode(const std::vector<vk::PresentModeKHR> availablePresentModes)
{
    vk::PresentModeKHR bestMode = vk::PresentModeKHR::eFifo;

    for(const auto& availablePresentMode : availablePresentModes)
    {
        if(availablePresentMode == vk::PresentModeKHR::eMailbox)
        {
            return availablePresentMode;
        }
        else if(availablePresentMode == vk::PresentModeKHR::eImmediate)
        {
            bestMode = availablePresentMode;
        }
    }

    return bestMode;
}

vk::Extent2D chooseSwapExtent(vk::Extent2D desiredExtent, const vk::SurfaceCapabilitiesKHR& capabilities)
{
    if(capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max())
    {
        return capabilities.currentExtent;
    }
    else
    {
        desiredExtent.width =
          std::max(capabilities.minImageExtent.width, std::min(capabilities.maxImageExtent.width, desiredExtent.width));
        desiredExtent.height = std::max(capabilities.minImageExtent.height,
                                        std::min(capabilities.maxImageExtent.height, desiredExtent.height));

        return desiredExtent;
    }
}

void Renderer::initSwapchain(vk::Extent2D desiredExtent)
{
    auto formats = m_gpu.getSurfaceFormatsKHR(m_surface.get());
    auto presentModes = m_gpu.getSurfacePresentModesKHR(m_surface.get());
    auto capabilities = m_gpu.getSurfaceCapabilitiesKHR(m_surface.get());

    vk::SurfaceFormatKHR surfaceFormat = chooseSwapSurfaceFormat(formats);
    vk::PresentModeKHR presentMode = chooseSwapPresentMode(presentModes);
    vk::Extent2D extent = chooseSwapExtent(desiredExtent, capabilities);

    uint32_t imageCount = capabilities.minImageCount + 1;
    if(capabilities.maxImageCount > 0 && imageCount > capabilities.maxImageCount)
    {
        imageCount = capabilities.maxImageCount;
    }

    auto swapCreateInfo = vk::SwapchainCreateInfoKHR();
    swapCreateInfo.setMinImageCount(imageCount);
    swapCreateInfo.setImageFormat(surfaceFormat.format);
    swapCreateInfo.setImageColorSpace(surfaceFormat.colorSpace);
    swapCreateInfo.setImageExtent(extent);
    swapCreateInfo.setImageArrayLayers(1);
    swapCreateInfo.setImageUsage(vk::ImageUsageFlagBits::eColorAttachment);
    swapCreateInfo.setPreTransform(capabilities.currentTransform);
    swapCreateInfo.setCompositeAlpha(vk::CompositeAlphaFlagBitsKHR::eOpaque);
    swapCreateInfo.setPresentMode(presentMode);
    swapCreateInfo.setClipped(true);
    swapCreateInfo.setOldSwapchain(m_swapchain.get());
    swapCreateInfo.setSurface(m_surface.get());

    uint32_t queueFamilyIndices[] = { m_graphicsFamilyIdx, m_presentFamilyIdx };
    if(m_graphicsFamilyIdx != m_presentFamilyIdx)
    {
        swapCreateInfo.setImageSharingMode(vk::SharingMode::eConcurrent);
        swapCreateInfo.setQueueFamilyIndexCount(2);
        swapCreateInfo.setPQueueFamilyIndices(queueFamilyIndices);
    }
    else
    {
        swapCreateInfo.setImageSharingMode(vk::SharingMode::eExclusive);
        swapCreateInfo.setQueueFamilyIndexCount(0);
        swapCreateInfo.setPQueueFamilyIndices(nullptr);
    }

    try
    {
        m_swapchain = m_device->createSwapchainKHRUnique(swapCreateInfo);
    }
    catch(std::exception e)
    {
        std::cout << "Could not create a Vulkan swapchain: " << e.what() << std::endl;
        throw;
    }

    try
    {
        m_swapchainImages = m_device->getSwapchainImagesKHR(m_swapchain.get());
    }
    catch(std::exception e)
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

    for(const auto& image : m_swapchainImages)
    {
        auto imageViewCreateInfo = vk::ImageViewCreateInfo();
        imageViewCreateInfo.setViewType(vk::ImageViewType::e2D);
        imageViewCreateInfo.setFormat(m_swapchainFormat);
        imageViewCreateInfo.setComponents(
            vk::ComponentMapping(
                vk::ComponentSwizzle::eIdentity,
                vk::ComponentSwizzle::eIdentity,
                vk::ComponentSwizzle::eIdentity,
                vk::ComponentSwizzle::eIdentity));
        imageViewCreateInfo.setSubresourceRange(
            vk::ImageSubresourceRange()
            .setAspectMask(vk::ImageAspectFlagBits::eColor)
            .setBaseArrayLayer(0)
            .setBaseMipLevel(0)
            .setLayerCount(1)
            .setLevelCount(1));
        imageViewCreateInfo.setImage(image);

        try
        {
            m_swapChainImageViews.push_back(m_device->createImageViewUnique(imageViewCreateInfo));
        }
        catch(std::exception e)
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
}

void Renderer::beginRender()
{
    m_device->waitForFences(m_inFlightFences[m_currentFrameIndex].get(), true, UINT64_MAX);
    m_device->resetFences(m_inFlightFences[m_currentFrameIndex].get());

    auto imageIndex = m_device->acquireNextImageKHR(m_swapchain.get(), UINT64_MAX, m_imageAvailableSemaphores[m_currentFrameIndex].get(), nullptr);

    if(imageIndex.result != vk::Result::eSuccess)
    {
        throw std::exception("Failed to prepair rendering");
    }

    m_currentImageIndex = imageIndex.value;

    const auto cbbi = vk::CommandBufferBeginInfo()
                        .setFlags(vk::CommandBufferUsageFlagBits::eSimultaneousUse);

    m_commandBuffers[m_currentImageIndex]->begin(cbbi);

    const auto clearValue = vk::ClearValue(vk::ClearColorValue(std::array<float, 4>{0.f, 0.f, 0.f, 0.f}));
    const auto rpbi = vk::RenderPassBeginInfo()
                        .setRenderPass(m_renderPass.get())
                        .setFramebuffer(m_framebuffers[m_currentImageIndex].get())
                        .setRenderArea(vk::Rect2D({ 0, 0 }, m_swapchainExtent))
                        .setClearValueCount(1)
                        .setPClearValues(&clearValue);

    m_commandBuffers[m_currentImageIndex]->beginRenderPass(rpbi, vk::SubpassContents::eInline);

    m_commandBuffers[m_currentImageIndex]->bindPipeline(vk::PipelineBindPoint::eGraphics, m_graphicsPipeline.get());
}

void Renderer::endRender()
{
    m_commandBuffers[m_currentImageIndex]->endRenderPass();
    m_commandBuffers[m_currentImageIndex]->end();

    std::vector<vk::PipelineStageFlags> waitStages = {vk::PipelineStageFlagBits::eColorAttachmentOutput};

    std::vector<vk::Semaphore> waitSemaphores = {m_imageAvailableSemaphores[m_currentFrameIndex].get()};
    std::vector<vk::Semaphore> signalSemaphores = {m_renderFinishedSemaphores[m_currentFrameIndex].get()};
    std::vector<vk::CommandBuffer> submitCommandBuffer = {m_commandBuffers[m_currentImageIndex].get()};

    auto submitInfo = vk::SubmitInfo();
    submitInfo.setPCommandBuffers(submitCommandBuffer.data());
    submitInfo.setCommandBufferCount(static_cast<uint32_t>(submitCommandBuffer.size()));
    submitInfo.setPWaitSemaphores(waitSemaphores.data());
    submitInfo.setWaitSemaphoreCount(static_cast<uint32_t>(waitSemaphores.size()));
    submitInfo.setPWaitDstStageMask(waitStages.data());
    submitInfo.setPSignalSemaphores(signalSemaphores.data());
    submitInfo.setSignalSemaphoreCount(static_cast<uint32_t>(signalSemaphores.size()));

    m_queue.submit(submitInfo, m_inFlightFences[m_currentImageIndex].get());
}

void Renderer::present()
{
    auto presentInfo = vk::PresentInfoKHR();
    presentInfo.setSwapchainCount(1);
    presentInfo.setPSwapchains(&m_swapchain.get());
    presentInfo.setPImageIndices(&m_currentImageIndex);
    presentInfo.setPWaitSemaphores(&m_renderFinishedSemaphores[m_currentFrameIndex].get());
    presentInfo.setWaitSemaphoreCount(1);
    m_presentQueue.presentKHR(presentInfo);
    m_currentFrameIndex = (m_currentFrameIndex + 1) % m_inFlightFences.size();
}

void Renderer::waitForIdle()
{
    m_device->waitIdle();
}

void Renderer::initRenderPass()
{
    auto colorAttachment = vk::AttachmentDescription();
    colorAttachment.setFormat(m_swapchainFormat);
    colorAttachment.setSamples(vk::SampleCountFlagBits::e1);
    colorAttachment.setLoadOp(vk::AttachmentLoadOp::eClear);
    colorAttachment.setStoreOp(vk::AttachmentStoreOp::eStore);
    colorAttachment.setStencilLoadOp(vk::AttachmentLoadOp::eDontCare);
    colorAttachment.setStencilStoreOp(vk::AttachmentStoreOp::eDontCare);
    colorAttachment.setInitialLayout(vk::ImageLayout::eUndefined);
    colorAttachment.setFinalLayout(vk::ImageLayout::ePresentSrcKHR);

    auto colorAttachmentRef = vk::AttachmentReference();
    colorAttachmentRef.setAttachment(0);
    colorAttachmentRef.setLayout(vk::ImageLayout::eColorAttachmentOptimal);

    auto subpass = vk::SubpassDescription();
    subpass.setPipelineBindPoint(vk::PipelineBindPoint::eGraphics);
    subpass.setColorAttachmentCount(1);
    subpass.setPColorAttachments(&colorAttachmentRef);

    auto dependency = vk::SubpassDependency();
    dependency.setSrcSubpass(VK_SUBPASS_EXTERNAL);
    dependency.setDstSubpass(0);
    dependency.setSrcStageMask(vk::PipelineStageFlagBits::eColorAttachmentOutput);
    dependency.setSrcAccessMask(vk::AccessFlags());
    dependency.setDstStageMask(vk::PipelineStageFlagBits::eColorAttachmentOutput);
    dependency.setDstAccessMask(vk::AccessFlagBits::eColorAttachmentRead | vk::AccessFlagBits::eColorAttachmentWrite);

    auto renderPassCreateInfo = vk::RenderPassCreateInfo();
    renderPassCreateInfo.setAttachmentCount(1);
    renderPassCreateInfo.setPAttachments(&colorAttachment);
    renderPassCreateInfo.setSubpassCount(1);
    renderPassCreateInfo.setPSubpasses(&subpass);
    renderPassCreateInfo.setPDependencies(&dependency);
    renderPassCreateInfo.setDependencyCount(1);

    try
    {
        m_renderPass = m_device->createRenderPassUnique(renderPassCreateInfo);
    }
    catch(std::exception e)
    {
        std::cout << "Could not create a Vulkan render pass: " << e.what() << std::endl;
        throw;
    }
}

void Renderer::initPipelineCache()
{
    auto pipelineCacheCreateInfo = vk::PipelineCacheCreateInfo();
    pipelineCacheCreateInfo.setInitialDataSize(0U);
    pipelineCacheCreateInfo.setPInitialData(nullptr);

    try
    {
        m_pipelineCache = m_device->createPipelineCacheUnique(pipelineCacheCreateInfo);
    }
    catch(std::exception e)
    {
        std::cout << "Could not create a Vulkan pipelines cache: " << e.what() << std::endl;
        throw;
    }
}

void Renderer::initPipeline(const Material& mat)
{
    auto[vertShader, fragShader] = mat.getShaders();

    auto vertShaderStageInfo = vk::PipelineShaderStageCreateInfo();
    vertShaderStageInfo.setStage(vk::ShaderStageFlagBits::eVertex);
    vertShaderStageInfo.setModule(vertShader);
    vertShaderStageInfo.setPName("main");

    auto fragShaderBinStageInfo = vk::PipelineShaderStageCreateInfo();
    fragShaderBinStageInfo.setStage(vk::ShaderStageFlagBits::eFragment);
    fragShaderBinStageInfo.setModule(fragShader);
    fragShaderBinStageInfo.setPName("main");

    auto shaderStages = std::vector<vk::PipelineShaderStageCreateInfo>{ vertShaderStageInfo, fragShaderBinStageInfo };

    auto vertexInputInfo = vk::PipelineVertexInputStateCreateInfo();
    vertexInputInfo.setVertexBindingDescriptionCount(0);
    vertexInputInfo.setVertexAttributeDescriptionCount(0);

    auto inputAssembly = vk::PipelineInputAssemblyStateCreateInfo();
    inputAssembly.setTopology(vk::PrimitiveTopology::eTriangleList);
    inputAssembly.setPrimitiveRestartEnable(false);

    auto viewport = vk::Viewport();
    viewport.setX(0.0f);
    viewport.setY(0.0f);
    viewport.setWidth(static_cast<float>(m_swapchainExtent.width));
    viewport.setHeight(static_cast<float>(m_swapchainExtent.height));
    viewport.setMinDepth(0.0f);
    viewport.setMaxDepth(1.0f);

    auto scissor = vk::Rect2D();
    scissor.setOffset({ 0, 0 });
    scissor.setExtent(m_swapchainExtent);

    auto viewportState = vk::PipelineViewportStateCreateInfo();
    viewportState.setViewportCount(1);
    viewportState.setPViewports(&viewport);
    viewportState.setScissorCount(1);
    viewportState.setPScissors(&scissor);

    auto rasterizer = vk::PipelineRasterizationStateCreateInfo();
    rasterizer.setDepthClampEnable(false);
    rasterizer.setRasterizerDiscardEnable(false);
    rasterizer.setPolygonMode(vk::PolygonMode::eFill);
    rasterizer.setLineWidth(1.0f);
    rasterizer.setCullMode(vk::CullModeFlagBits::eBack);
    rasterizer.setFrontFace(vk::FrontFace::eClockwise);
    rasterizer.setDepthBiasEnable(true);

    auto multisampling = vk::PipelineMultisampleStateCreateInfo();
    multisampling.setSampleShadingEnable(false);
    multisampling.setRasterizationSamples(vk::SampleCountFlagBits::e1);

    auto colorBlendAttachment = vk::PipelineColorBlendAttachmentState();
    colorBlendAttachment.colorWriteMask =
        vk::ColorComponentFlagBits::eR |
        vk::ColorComponentFlagBits::eG |
        vk::ColorComponentFlagBits::eB |
        vk::ColorComponentFlagBits::eA;
    colorBlendAttachment.blendEnable = false;

    auto colorBlending = vk::PipelineColorBlendStateCreateInfo();
    colorBlending.setLogicOpEnable(false);
    colorBlending.setLogicOp(vk::LogicOp::eCopy);
    colorBlending.setAttachmentCount(1);
    colorBlending.setPAttachments(&colorBlendAttachment);
    colorBlending.setBlendConstants({ { 0.0f, 0.0f, 0.0f, 0.0f } });

    auto pipelineLayoutCreateInfo = vk::PipelineLayoutCreateInfo();
    pipelineLayoutCreateInfo.setSetLayoutCount(0U);
    pipelineLayoutCreateInfo.setPushConstantRangeCount(0U);

    try
    {
        m_pipelineLayout = m_device->createPipelineLayoutUnique(pipelineLayoutCreateInfo);
    }
    catch(std::exception e)
    {
        std::cout << "Could not create a Vulkan pipeline layout: " << e.what() << std::endl;
        throw;
    }

    auto pipelineCreateInfo = vk::GraphicsPipelineCreateInfo();
    pipelineCreateInfo.setStageCount(static_cast<uint32_t>(shaderStages.size()));
    pipelineCreateInfo.setPStages(shaderStages.data());
    pipelineCreateInfo.setPVertexInputState(&vertexInputInfo);
    pipelineCreateInfo.setPInputAssemblyState(&inputAssembly);
    pipelineCreateInfo.setPViewportState(&viewportState);
    pipelineCreateInfo.setPRasterizationState(&rasterizer);
    pipelineCreateInfo.setPMultisampleState(&multisampling);
    pipelineCreateInfo.setPColorBlendState(&colorBlending);
    pipelineCreateInfo.setLayout(m_pipelineLayout.get());
    pipelineCreateInfo.setRenderPass(m_renderPass.get());
    pipelineCreateInfo.setSubpass(0);
    pipelineCreateInfo.setBasePipelineHandle(nullptr);

    try
    {
        m_graphicsPipeline = m_device->createGraphicsPipelineUnique(m_pipelineCache.get(), pipelineCreateInfo);
    }
    catch(std::exception e)
    {
        std::cout << "Could not create a Vulkan graphics pipeline: " << e.what() << std::endl;
        throw;
    }
}

void Renderer::initFramebuffers()
{
    m_framebuffers.resize(m_swapChainImageViews.size());
    for(size_t i = 0; i < m_swapChainImageViews.size(); i++)
    {
        std::vector<vk::ImageView> attachments = { m_swapChainImageViews[i].get() };

        auto framebufferCreateInfo = vk::FramebufferCreateInfo();
        framebufferCreateInfo.setRenderPass(m_renderPass.get());
        framebufferCreateInfo.setAttachmentCount(static_cast<uint32_t>(attachments.size()));
        framebufferCreateInfo.setPAttachments(attachments.data());
        framebufferCreateInfo.setWidth(m_swapchainExtent.width);
        framebufferCreateInfo.setHeight(m_swapchainExtent.height);
        framebufferCreateInfo.setLayers(1);

        try
        {
            m_framebuffers[i] = m_device->createFramebufferUnique(framebufferCreateInfo);
        }
        catch(std::exception e)
        {
            std::cout << "Could not create a Vulkan framebuffer: " << e.what() << std::endl;
            throw;
        }
    }
}

void Renderer::initCommandPool()
{
    auto cpci = vk::CommandPoolCreateInfo();
    cpci.setQueueFamilyIndex(m_graphicsFamilyIdx);
    cpci.setFlags(vk::CommandPoolCreateFlagBits::eResetCommandBuffer);

    m_commandPool = m_device->createCommandPoolUnique(cpci);
}

void Renderer::initCommandBuffers()
{
    auto cbci = vk::CommandBufferAllocateInfo();
    cbci.setCommandPool(m_commandPool.get());
    cbci.setCommandBufferCount(static_cast<uint32_t>(m_framebuffers.size()));
    cbci.setLevel(vk::CommandBufferLevel::ePrimary);

    m_commandBuffers = m_device->allocateCommandBuffersUnique(cbci);
}

void Renderer::initSyncObjects()
{
    const auto sci = vk::SemaphoreCreateInfo();
    auto fci = vk::FenceCreateInfo(vk::FenceCreateFlagBits::eSignaled);
    for(size_t i = 0; i < 3; ++i)
    {
        m_imageAvailableSemaphores.push_back(m_device->createSemaphoreUnique(sci));
        m_renderFinishedSemaphores.push_back( m_device->createSemaphoreUnique(sci));
        m_inFlightFences.push_back(m_device->createFenceUnique(fci));
    }
}
