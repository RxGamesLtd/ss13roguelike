#pragma once

#include "vulkan/vulkan.hpp"

#include <string>

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

private:
    const Renderer& m_renderer;
    const std::string m_shaderName;
    vk::UniqueShaderModule m_vertexShader;
    vk::UniqueShaderModule m_fragmentShader;
};
