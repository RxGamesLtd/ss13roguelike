#pragma once

namespace Config
{
constexpr bool isDebug =
#if defined(DEBUG_VULKAN)
    true;
#else
    false;
#endif

}
