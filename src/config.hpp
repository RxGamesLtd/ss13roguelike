#pragma once

namespace Config
{
constexpr bool isDebug =
#if defined(_DEBUG)
    true;
#else
    false;
#endif


}
