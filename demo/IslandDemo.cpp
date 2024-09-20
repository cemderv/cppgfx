#include "IslandDemo.hpp"

#include "Camera.hpp"
#include "Content.hpp"

static auto ToRgbaf(const glm::vec4& v) -> ras::rgba32f
{
  return {v.r, v.g, v.b, v.a};
}

class IslandVS
{
public:
  glm::mat4 World;
  glm::mat4 View;
  glm::mat4 Projection;
  glm::mat4 ViewProjection;
  glm::mat4 WorldViewProj;

  auto execute(const MeshVertex& in) const -> IslandDemo::IslandVSOutput
  {
    const auto posCs  = WorldViewProj * glm::vec4{in.pos, 1.0F};
    const auto normal = World * glm::vec4{in.normal, 0.0F};

    return {
        .sv_pos = posCs,
        .Normal = glm::normalize(normal),
        .UV     = in.uv,
        .Color  = in.color,
    };
  }
};

template <bool HighQuality>
class IslandPS
{
public:
  auto execute(const IslandDemo::IslandVSOutput& in) const -> std::array<ras::rgba32f, 1>
  {
    const auto n     = glm::normalize(in.Normal);
    const auto l     = -glm::vec3{-0.5f, -1.0F, -1.0F};
    auto       ndotl = glm::dot(n, l);
    ndotl *= 0.5f;
    ndotl += 0.75f;
    ndotl = std::sqrt(ndotl);

    return {
        ToRgbaf({in.Color.rgb() * ndotl, in.Color.w}),
    };
  }
};

class RiverVS
{
public:
  glm::mat4 World;
  glm::mat4 View;
  glm::mat4 Projection;
  glm::mat4 ViewProjection;
  glm::mat4 WorldViewProjection;
  float     Time{};
  float     AnimX{};
  float     AnimY{};
  float     UVOffsetX{};

  auto execute(const MeshVertex& in) const -> IslandDemo::RiverVSOutput
  {
    auto inPos = in.pos;
    inPos.x += AnimX;
    inPos.y += AnimY;

    const auto posCs = WorldViewProjection * glm::vec4{inPos, 1.0F};
    const auto uv    = (in.uv * 1.7F) + glm::vec2{UVOffsetX, 0.0F};

    return {
        .sv_pos = posCs,
        .UV     = uv,
    };
  }
};

class RiverPS
{
public:
  auto execute(const IslandDemo::RiverVSOutput& in) const -> std::array<ras::rgba32f, 1>
  {
    const ras::rgb32f color = ras::sample(*texture, in.UV.x, in.UV.y, ras::sampler_states::point_wrap).to_rgb();

    return {
        ras::rgba32f{color.r, color.g, color.b, 1.0F},
    };
  }

  const ras::image2d_rgba* texture;
};

class ParticleVS
{
public:
  glm::mat4 World;
  glm::mat4 WorldViewProjection;
  glm::vec2 ViewportScale;

  explicit ParticleVS(const ParticleSystem& sys)
      : m_System(sys)
  {
  }

  IslandDemo::ParticleVSOutput execute(const ParticleVertex& in) const
  {
    const float age           = in.age * (1.0F + in.randomValues.x * m_System.particleLifeTimeRandomness);
    const float normalizedAge = std::clamp(age / m_System.particleLifeTime, 0.0F, 1.0F);

    const glm::vec3 posOS = this->ComputeParticlePosition(in.position, in.velocity, age, normalizedAge);
    const float     size  = this->ComputeParticleSize(in.randomValues.y, normalizedAge);

    glm::vec4 posCS = WorldViewProjection * glm::vec4(posOS, 1.0F);

    const auto posOffset = in.corner * size * ViewportScale;
    posCS.x += posOffset.x;
    posCS.y += posOffset.y;

    return {
        .sv_pos = posCS,
        .Color  = this->ComputeParticleColor(in.randomValues.z, normalizedAge),
    };
  }

private:
  auto ComputeParticlePosition(const glm::vec3& position,
                               const glm::vec3& velocity,
                               float            age,
                               float            normalizedAge) const -> glm::vec3
  {
    const float startVelocity = glm::length(velocity);
    const float endVelocityL  = startVelocity * m_System.endVelocity;

    const float velocityIntegral =
        startVelocity * normalizedAge + (endVelocityL - startVelocity) * normalizedAge * normalizedAge * 0.5F;

    glm::vec3 outPos = position;
    outPos += glm::normalize(velocity) * velocityIntegral * m_System.particleLifeTime;
    outPos += m_System.gravity * age * normalizedAge;

    return outPos;
  }

  auto ComputeParticleSize(float randomValue, float normalizedAge) const -> float
  {
    const float startSizeL = std::lerp(m_System.startSize.x, m_System.startSize.y, randomValue);
    const float endSizeL   = std::lerp(m_System.endSize.x, m_System.endSize.y, randomValue);
    const float size       = std::lerp(startSizeL, endSizeL, normalizedAge);
    return size * 5.0F;
  }

  auto ComputeParticleColor(float randomValue, float normalizedAge) const -> glm::vec4
  {
    const float invAge = 1.0F - normalizedAge;

    glm::vec4 color = glm::mix(m_System.minColor, m_System.maxColor, randomValue);
    color.w *= normalizedAge * invAge * 6.7F;

    return color;
  }

  const ParticleSystem& m_System;
};

class ParticlePS
{
public:
  std::array<ras::rgba32f, 1> execute(const IslandDemo::ParticleVSOutput& in) const
  {
    return {
        ToRgbaf(in.Color),
    };
  }
};

IslandDemo::IslandDemo()
    : m_SnowParticleEmissionCounter()
    , m_FireParticleEmissionCounter()
    , m_RiverUvTime(0)
    , m_IsCameraMouseDown(false)
{
  m_SnowParticles.particleLifeTime   = 3.5F;
  m_SnowParticles.startSize          = glm::vec2(0.08f, 0.08f);
  m_SnowParticles.endSize            = glm::vec2(0.04f, 0.04f);
  m_SnowParticles.minColor           = glm::vec4(1, 1, 1, 1);
  m_SnowParticles.maxColor           = glm::vec4(1, 1, 1, 0);
  m_SnowParticles.horizontalVelocity = glm::vec2(0.2f, 0.4f);
  m_SnowParticles.verticalVelocity   = glm::vec2(0.0F, 0.1f);
  m_SnowParticles.gravity            = {0, 0, 0};

  m_FireParticles.particleLifeTime   = 1.5F;
  m_FireParticles.startSize          = glm::vec2(0.08f, 0.08f);
  m_FireParticles.endSize            = glm::vec2(0.04f, 0.04f);
  m_FireParticles.minColor           = glm::vec4(1, 1, 0, 1);
  m_FireParticles.maxColor           = glm::vec4(1, 0, 0, 0.4f);
  m_FireParticles.horizontalVelocity = glm::vec2(-0.1f, 0.1f);
  m_FireParticles.verticalVelocity   = glm::vec2(0.0F, 0.0F);
  m_FireParticles.endVelocity        = 0.0F;
  m_FireParticles.gravity            = {0, 0.3f, 0};

  m_Camera.Target       = glm::vec3{0, 0, 0};
  m_Camera.Yaw          = glm::radians(140.0F);
  m_Camera.Pitch        = glm::radians(5.0F);
  m_Camera.PitchRange   = {1.0F, 15.0F};
  m_Camera.ZoomDistance = 12.0F;
  m_Camera.ZoomRange    = {3.0F, 12.0F};

  m_IslandMesh   = content::LoadMeshFromFile("Assets/Island.glb");
  m_RiverMesh    = content::LoadMeshFromFile("Assets/River.glb");
  m_RiverTexture = content::LoadImageFromFile("Assets/River.png");
}

void IslandDemo::Update(float elapsedTime)
{
  m_RiverUvTime += elapsedTime;

  m_SnowParticles.update(elapsedTime);
  m_FireParticles.update(elapsedTime);

  auto randomXDist = std::uniform_real_distribution{-2.0F, 5.0F};
  auto randomYDist = std::uniform_real_distribution{-4.0F, 4.0F};

  m_SnowParticleEmissionCounter += elapsedTime;

  if (m_SnowParticleEmissionCounter > 0.1f)
  {
    for (int i = 0; i < 15; ++i)
    {
      const auto x = randomXDist(m_Random);
      const auto z = randomYDist(m_Random);

      m_SnowParticles.emitParticle({x, 2.0F, z}, glm::vec3(-0.2f, -1, -0.2f));
    }

    m_SnowParticleEmissionCounter = 0.0F;
  }

  m_FireParticleEmissionCounter += elapsedTime;

  if (m_FireParticleEmissionCounter > 0.05f)
  {
    m_FireParticles.emitParticle({2.27F, -0.45F, -1.685f}, glm::vec3(0, 0.01F, 0));
    m_FireParticleEmissionCounter = 0.0F;
  }
}

void IslandDemo::Draw(ras::image2d_bgra& backBuffer, ras::image2d_depth32& depthBuffer)
{
  backBuffer.clear(ras::b8g8r8a8{uint8_t(165), uint8_t(212), uint8_t(232), uint8_t(255)});
  depthBuffer.clear(1.0F);

  const auto viewMat     = m_Camera.GetViewMatrix();
  const auto projMat     = m_Camera.GetProjectionMatrix(int(backBuffer.width()), int(backBuffer.height()));
  const auto viewProjMat = projMat * viewMat;

  this->DrawIsland(backBuffer, depthBuffer, viewMat, projMat, viewProjMat);
  this->DrawRiver(backBuffer, depthBuffer, viewMat, projMat, viewProjMat);
  this->DrawSnowParticles(backBuffer, depthBuffer, projMat, viewProjMat);
  this->DrawFireParticles(backBuffer, depthBuffer, projMat, viewProjMat);
}

void IslandDemo::MouseHasMoved(glm::vec2 delta)
{
  if (m_IsCameraMouseDown)
  {
    m_Camera.Pitch += delta.y * m_Camera.RotationSensitivity;
    m_Camera.Yaw -= delta.x * m_Camera.RotationSensitivity;

    const auto [pitchMin, pitchMax] = m_Camera.PitchRange;

    m_Camera.Pitch = std::clamp(m_Camera.Pitch, glm::radians(pitchMin), glm::radians(pitchMax));
  }
}

void IslandDemo::DrawIsland(ras::image2d_bgra&    backBuffer,
                            ras::image2d_depth32& depthBuffer,
                            const glm::mat4&      viewMat,
                            const glm::mat4&      projMat,
                            const glm::mat4&      viewProjMat)
{
  IslandVS vs;

  vs.World          = glm::mat4{1.0F};
  vs.View           = viewMat;
  vs.Projection     = projMat;
  vs.ViewProjection = viewProjMat;
  vs.WorldViewProj  = vs.ViewProjection * vs.World;

  IslandPS<false> ps;

  const auto vprect = ras::rect{0, 0, backBuffer.width(), backBuffer.height()};
  const auto vp     = ras::viewport{vprect, 0.0F, 1.0F};

  const auto& submesh = m_IslandMesh.submeshes.front();

  ras::draw_indexed(ras::color_images(backBuffer),
                    depthBuffer,
                    vs,
                    ps,
                    vp,
                    vprect,
                    ras::blend_states::opaque,
                    ras::depth_stencil_states::depth_default,
                    ras::rasterizer_states::cull_clockwise,
                    ras::primitive_topology::triangle_list,
                    m_IslandRasterCache,
                    ras::vertex_view(m_IslandMesh.vertices),
                    ras::index_view(m_IslandMesh.indices),
                    submesh.startIndex,
                    submesh.indexCount);
}

void IslandDemo::DrawRiver(ras::image2d_bgra&    backBuffer,
                           ras::image2d_depth32& depthBuffer,
                           const glm::mat4&      viewMat,
                           const glm::mat4&      projMat,
                           const glm::mat4&      viewProjMat)
{
  RiverVS vs;
  vs.World               = glm::mat4{1.0F};
  vs.View                = viewMat;
  vs.Projection          = projMat;
  vs.ViewProjection      = viewProjMat;
  vs.WorldViewProjection = vs.ViewProjection * vs.World;
  vs.Time                = float(m_RiverUvTime);
  vs.AnimX += std::cos(vs.Time * 2.8f) * 0.024f;
  vs.AnimY += std::sin(vs.Time * 3.0f) * 0.01f;
  vs.UVOffsetX = vs.Time * 1.5f;

  RiverPS ps;
  ps.texture = &m_RiverTexture;

  const auto vprect = ras::rect(0, 0, backBuffer.width(), backBuffer.height());
  const auto vp     = ras::viewport(vprect, 0.0F, 1.0F);

  const auto& submesh = m_RiverMesh.submeshes[0];

  ras::draw_indexed(ras::color_images(backBuffer),
                    depthBuffer,
                    vs,
                    ps,
                    vp,
                    vprect,
                    ras::blend_states::additive,
                    ras::depth_stencil_states::depth_read,
                    ras::rasterizer_states::cull_none,
                    ras::primitive_topology::triangle_list,
                    m_RiverRasterCache,
                    ras::vertex_view(m_RiverMesh.vertices),
                    ras::index_view(m_RiverMesh.indices),
                    submesh.startIndex,
                    submesh.indexCount);
}

void IslandDemo::DrawSnowParticles(ras::image2d_bgra&    backBuffer,
                                   ras::image2d_depth32& depthBuffer,
                                   const glm::mat4&      projMat,
                                   const glm::mat4&      viewProjMat)
{
  std::ignore = projMat;

  ParticleVS vs{m_SnowParticles};
  vs.World               = glm::mat4{1.0F};
  vs.WorldViewProjection = viewProjMat * vs.World;
  vs.ViewportScale       = glm::vec2{0.5F / backBuffer.aspect_ratio(), -0.5F};

  ParticlePS ps;

  const auto vprect = ras::rect(0, 0, backBuffer.width(), backBuffer.height());
  const auto vp     = ras::viewport(vprect, 0.0F, 1.0F);

  m_SnowParticles.draw([&](const ras::vertex_view<ParticleVertex>& vertices,
                           const ras::index_view<uint16_t>&        indices,
                           size_t                                  startIndex,
                           size_t                                  indexCount) {
    ras::draw_indexed(ras::color_images(backBuffer),
                      depthBuffer,
                      vs,
                      ps,
                      vp,
                      vprect,
                      ras::blend_states::non_premultiplied,
                      ras::depth_stencil_states::depth_read,
                      ras::rasterizer_states::cull_counter_clockwise,
                      ras::primitive_topology::triangle_list,
                      m_ParticleRasterCache,
                      vertices,
                      indices,
                      startIndex,
                      indexCount);
  });
}

void IslandDemo::DrawFireParticles(ras::image2d_bgra&    backBuffer,
                                   ras::image2d_depth32& depthBuffer,
                                   const glm::mat4&      projMat,
                                   const glm::mat4&      viewProjMat)
{
  std::ignore = projMat;

  ParticleVS vs{m_FireParticles};
  vs.World               = glm::mat4{1.0F};
  vs.WorldViewProjection = viewProjMat * vs.World;
  vs.ViewportScale       = glm::vec2{0.5F / backBuffer.aspect_ratio(), -0.5F};

  ParticlePS ps;

  const auto vprect = ras::rect(0, 0, backBuffer.width(), backBuffer.height());
  const auto vp     = ras::viewport(vprect, 0.0F, 1.0F);

  m_FireParticles.draw([&](const ras::vertex_view<ParticleVertex>& vertices,
                           const ras::index_view<uint16_t>&        indices,
                           size_t                                  startIndex,
                           size_t                                  indexCount) {
    ras::draw_indexed(ras::color_images(backBuffer),
                      depthBuffer,
                      vs,
                      ps,
                      vp,
                      vprect,
                      ras::blend_states::non_premultiplied,
                      ras::depth_stencil_states::depth_read,
                      ras::rasterizer_states::cull_counter_clockwise,
                      ras::primitive_topology::triangle_list,
                      m_ParticleRasterCache,
                      vertices,
                      indices,
                      startIndex,
                      indexCount);
  });
}

void IslandDemo::IsMouseButtonDown(int button)
{
  std::ignore         = button;
  m_IsCameraMouseDown = true;
}

void IslandDemo::IsMouseButtonUp(int button)
{
  std::ignore         = button;
  m_IsCameraMouseDown = false;
}

void IslandDemo::MouseWheelHasMoved(glm::vec2 delta)
{
  m_Camera.ZoomDistance =
      std::clamp(m_Camera.ZoomDistance - delta.y * 0.25F, m_Camera.ZoomRange.first, m_Camera.ZoomRange.second);
}
