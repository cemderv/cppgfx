#pragma once

#include <cppgfx.hpp>
#include <string_view>

struct Mesh;

namespace content
{
  auto LoadMeshFromFile(std::string_view name) -> Mesh;

  auto LoadImageFromFile(std::string_view name) -> ras::image2d_rgba;
} // namespace content
