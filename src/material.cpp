#include "material.hpp"

#include "renderer.hpp"
#include "shaderc/shaderc.hpp"

#include <fstream>
#include <iostream>

static std::string readFile(const std::string& filename)
{
    std::ifstream file(filename, std::ios::ate | std::ios::binary);

    if(!file.is_open())
    {
        throw std::runtime_error("failed to open file!");
    }

    size_t fileSize = static_cast<size_t>(file.tellg());
    std::vector<char> buffer(fileSize);

    file.seekg(0);
    file.read(buffer.data(), fileSize);

    file.close();

    return { buffer.begin(), buffer.end() };
}

Material::Material(const Renderer& render, const std::string& shaderName)
    : m_renderer(render)
    , m_shaderName(shaderName)
{
    compileShaders();
}

vk::UniqueShaderModule Material::createShaderModule(const vk::Device& device, const std::vector<uint32_t>& code) const
{
    auto createInfo = vk::ShaderModuleCreateInfo()
                        .setCodeSize(code.size() * sizeof(uint32_t))
                        .setPCode(code.data());

    return device.createShaderModuleUnique(createInfo);
}

void Material::compileShaders()
{
    auto vertShaderCode = readFile(std::string("src/shaders/") + m_shaderName + std::string(".vert"));
    auto fragShaderCode = readFile(std::string("src/shaders/") + m_shaderName + std::string(".frag"));

    shaderc::Compiler shaderCompiler;
    shaderc::CompileOptions shaderCompilerOptions;
    shaderCompilerOptions.SetOptimizationLevel(shaderc_optimization_level_size);
    if(!shaderCompiler.IsValid())
    {
        throw std::runtime_error("Failed to initilize shader compiler.");
    }

    auto vertShaderBin = shaderCompiler.CompileGlslToSpv(
        vertShaderCode,
        shaderc_shader_kind::shaderc_glsl_vertex_shader,
        m_shaderName.c_str(),
        shaderCompilerOptions);
    auto fragShaderBin = shaderCompiler.CompileGlslToSpv(
        fragShaderCode,
        shaderc_shader_kind::shaderc_glsl_fragment_shader,
        m_shaderName.c_str(),
        shaderCompilerOptions);

    if(vertShaderBin.GetCompilationStatus() != shaderc_compilation_status_success)
    {
        std::cout << "Vert shader error:" << vertShaderBin.GetErrorMessage() << std::endl;
    }
    if(fragShaderBin.GetCompilationStatus() != shaderc_compilation_status_success)
    {
        std::cout << "Frag shader error:" << fragShaderBin.GetErrorMessage() << std::endl;
    }

    std::vector<uint32_t> vertData;
    std::vector<uint32_t> fragData;

    vertData.assign(vertShaderBin.cbegin(), vertShaderBin.cend());
    fragData.assign(fragShaderBin.cbegin(), fragShaderBin.cend());

    m_vertexShader = createShaderModule(m_renderer.getDevice(), vertData);
    m_fragmentShader = createShaderModule(m_renderer.getDevice(), fragData);
}

Shaders Material::getShaders() const
{
    return {m_vertexShader.get(), m_fragmentShader.get()};
}
