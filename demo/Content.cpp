#include "Content.hpp"
#include "Mesh.hpp"

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#define TINYGLTF_IMPLEMENTATION
#include <tiny_gltf.h>

#include <cmrc/cmrc.hpp>
CMRC_DECLARE(Demo);

namespace content
{
  static auto LoadMeshFromGltf(const tinygltf::TinyGLTF& gltf, const tinygltf::Model& model) -> Mesh
  {
    (void)gltf;

    Mesh mesh;
    mesh.vertices.reserve(256);
    mesh.indices.reserve(256);
    mesh.submeshes.reserve(model.meshes.size());

    for (const auto& gltfNode : model.nodes)
    {
      if (gltfNode.mesh < 0)
        continue;

      const auto& gltfMesh = model.meshes[gltfNode.mesh];

      Submesh submesh;
      submesh.name       = gltfMesh.name;
      submesh.startIndex = uint16_t(mesh.indices.size());

      const int startVertex = int(mesh.vertices.size());

      for (const auto& gltfPrim : gltfMesh.primitives)
      {
        if (gltfPrim.mode != TINYGLTF_MODE_TRIANGLES)
          continue;

        {
          const auto findAttribAccessor = [&](const std::string& name) {
            const auto it  = gltfPrim.attributes.find(name);
            const auto idx = it != gltfPrim.attributes.cend() ? it->second : -1;

            return idx >= 0 ? &model.accessors[idx] : nullptr;
          };

          const auto posAccessor    = findAttribAccessor("POSITION");
          const auto normalAccessor = findAttribAccessor("NORMAL");
          const auto uvAccessor     = findAttribAccessor("TEXCOORD_0");
          const auto colorAccessor  = findAttribAccessor("COLOR_0");

          const auto getBufferDataPtr = [&](const tinygltf::Accessor* accessor) -> const void* {
            if (accessor == nullptr)
              return nullptr;

            const auto& bufferView = model.bufferViews[accessor->bufferView];
            const auto& buffer     = model.buffers[bufferView.buffer];
            return buffer.data.data() + bufferView.byteOffset + accessor->byteOffset;
          };

          struct Vec4U16
          {
            uint16_t r;
            uint16_t g;
            uint16_t b;
            uint16_t a;

            glm::vec4 toVec4() const
            {
              const auto toFloat = [](uint16_t f) {
                const auto linear = float(double(f) / std::numeric_limits<uint16_t>::max());
                return std::pow(linear, 1.0F / 2.2F);
              };

              return {toFloat(r), toFloat(g), toFloat(b), toFloat(a)};
            }
          };

          const auto posData    = static_cast<const glm::vec3*>(getBufferDataPtr(posAccessor));
          const auto normalData = static_cast<const glm::vec3*>(getBufferDataPtr(normalAccessor));
          const auto uvData     = static_cast<const glm::vec2*>(getBufferDataPtr(uvAccessor));
          const auto colorData  = static_cast<const Vec4U16*>(getBufferDataPtr(colorAccessor));

          for (size_t i = 0; i < posAccessor->count; ++i)
          {
            MeshVertex vertex{};
            vertex.pos    = posData[i];
            vertex.normal = normalData != nullptr ? normalData[i] : glm::vec3{0.001f};
            vertex.uv     = uvData != nullptr ? uvData[i] : glm::vec2{0.3f};
            vertex.color  = glm::vec4{1, 0, 1, 1};
            vertex.color  = colorData != nullptr ? colorData[i].toVec4() : glm::vec4{1, 0, 1, 1};

            mesh.vertices.push_back(vertex);
          }
        }

        {
          const auto& indexAccessor = model.accessors[gltfPrim.indices];

          if (indexAccessor.componentType != TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT)
          {
            return {};
          }

          const auto& indexBufferView = model.bufferViews[indexAccessor.bufferView];
          const auto& indexBuffer     = model.buffers[indexBufferView.buffer];

          const auto indexData = reinterpret_cast<const uint16_t*>(
              indexBuffer.data.data() + indexBufferView.byteOffset + indexAccessor.byteOffset);

          for (size_t i = 0; i < indexAccessor.count; ++i) mesh.indices.push_back(uint16_t(indexData[i] + startVertex));
        }
      }

      submesh.indexCount = mesh.indices.size() - submesh.startIndex;

      mesh.submeshes.push_back(submesh);
    }

    assert(size_t(*std::max_element(mesh.indices.cbegin(), mesh.indices.cend())) < mesh.vertices.size());

    {
      glm::vec3 aabbMin{std::numeric_limits<float>::max()};
      glm::vec3 aabbMax{std::numeric_limits<float>::lowest()};

      for (const auto& vertex : mesh.vertices)
      {
        const auto& pos = vertex.pos;

        if (pos.x < aabbMin.x)
          aabbMin.x = pos.x;

        if (pos.y < aabbMin.y)
          aabbMin.y = pos.y;

        if (pos.z < aabbMin.z)
          aabbMin.z = pos.z;

        if (pos.x > aabbMax.x)
          aabbMax.x = pos.x;

        if (pos.y > aabbMax.y)
          aabbMax.y = pos.y;

        if (pos.z > aabbMax.z)
          aabbMax.z = pos.z;
      }

      mesh.aabbMin = aabbMin;
      mesh.aabbMax = aabbMax;
    }

    return mesh;
  }

  Mesh LoadMeshFromFile(std::string_view name)
  {
    auto fs  = cmrc::Demo::get_filesystem();
    auto ifs = fs.open(std::string{name});

    tinygltf::Model    model;
    std::string        error, warning;
    tinygltf::TinyGLTF gltf{};

    if (!gltf.LoadBinaryFromMemory(&model,
                                   &error,
                                   &warning,
                                   reinterpret_cast<const unsigned char*>(ifs.cbegin()),
                                   ifs.size()))
    {
      return {};
    }
    else
    {
      return LoadMeshFromGltf(gltf, model);
    }
  }

  ras::image2d_rgba LoadImageFromFile(std::string_view name)
  {
    auto fs  = cmrc::Demo::get_filesystem();
    auto ifs = fs.open(std::string{name});

    int        width, height, comp;
    const auto imageData = stbi_load_from_memory(reinterpret_cast<const stbi_uc*>(ifs.cbegin()),
                                                 int(ifs.size()),
                                                 &width,
                                                 &height,
                                                 &comp,
                                                 4);

    ras::image2d_rgba image{uint32_t(width), uint32_t(height)};
    std::memcpy(image.data(), imageData, image.size_in_bytes());

    stbi_image_free(imageData);

    return image;
  }
} // namespace content
