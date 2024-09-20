#pragma once

#include <glm/vec2.hpp>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>
#include <string>
#include <vector>

struct Submesh
{
  std::string name;
  size_t      startIndex{};
  size_t      indexCount{};
};

struct MeshVertex
{
  glm::vec3 pos;
  glm::vec3 normal;
  glm::vec2 uv;
  glm::vec4 color;
};

struct Mesh
{
  std::vector<MeshVertex> vertices;
  std::vector<uint16_t>   indices;
  std::vector<Submesh>    submeshes;
  glm::vec3               aabbMin;
  glm::vec3               aabbMax;
};
