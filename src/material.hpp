#pragma once

#include "vulkan/vulkan.hpp"

class Renderer;

struct Shaders
{
    vk::ShaderModule vertex;
    vk::ShaderModule fragment;
};

class Material
{
public:
    Material(const Renderer& render, const std::string& shaderName);

    Shaders getShaders() const;

protected:
    void compileShaders();
    vk::UniqueShaderModule createShaderModule(const vk::Device& device, const std::vector<uint32_t>& code) const;

private:
    const Renderer& m_renderer;
    const std::string m_shaderName;
    vk::UniqueShaderModule m_vertexShader;
    vk::UniqueShaderModule m_fragmentShader;
};
