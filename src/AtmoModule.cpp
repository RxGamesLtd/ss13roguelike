#include "stdafx.hpp"

#include "AtmoModule.hpp"

using namespace SS13Game;
using namespace Atmos;

template <size_t SizeX, size_t SizeY, size_t SizeZ>
SS13Game::Atmos::Grid<SizeX, SizeY, SizeZ>::Grid()
{
}

template <size_t SizeX, size_t SizeY, size_t SizeZ>
SS13Game::Atmos::Grid<SizeX, SizeY, SizeZ>::Grid(Grid<SizeX, SizeY, SizeZ>&& rhs)
{
}

template <size_t SizeX, size_t SizeY, size_t SizeZ>
Grid<SizeX, SizeY, SizeZ>& SS13Game::Atmos::Grid<SizeX, SizeY, SizeZ>::operator=(Grid<SizeX, SizeY, SizeZ>&& rhs)
{
}
