#pragma once
#include <array>

namespace SS13Game {
namespace Atmos {

template <size_t SizeX, size_t SizeY, size_t SizeZ>
class Grid
{
public:
    Grid();

    Grid(const Grid&) = delete;
    Grid& operator=(const Grid&) = delete;

    Grid(Grid&&);
    Grid& operator=(Grid&&);

protected:
    std::array<float, SizeX * SizeY * SizeZ> grid_;
};

} // namespace Atmos
} // namespace SS13Game
