#pragma once

#include "Camera.hpp"
#include "Demo.hpp"
#include "Mesh.hpp"
#include "ParticleSystem.hpp"
#include <random>

class IslandDemo : public Demo
{
public:
  struct IslandVSOutput
  {
    glm::vec4 sv_pos;
    glm::vec3 Normal;
    glm::vec2 UV;
    glm::vec4 Color;
  };

  struct ParticleVSOutput
  {
    glm::vec4 sv_pos;
    glm::vec4 Color;
  };

  struct RiverVSOutput
  {
    glm::vec4 sv_pos;
    glm::vec2 UV;
  };

  IslandDemo();

  void Update(float elapsedTime) override;

  void Draw(ras::image2d_bgra& backBuffer, ras::image2d_depth32& depthBuffer) override;

  void IsMouseButtonDown(int button) override;

  void IsMouseButtonUp(int button) override;

  void MouseHasMoved(glm::vec2 delta) override;

  void MouseWheelHasMoved(glm::vec2 delta) override;

private:
  void DrawIsland(ras::image2d_bgra&    backBuffer,
                  ras::image2d_depth32& depthBuffer,
                  const glm::mat4&      viewMat,
                  const glm::mat4&      projMat,
                  const glm::mat4&      viewProjMat);

  void DrawRiver(ras::image2d_bgra&    backBuffer,
                 ras::image2d_depth32& depthBuffer,
                 const glm::mat4&      viewMat,
                 const glm::mat4&      projMat,
                 const glm::mat4&      viewProjMat);

  void DrawSnowParticles(ras::image2d_bgra&    backBuffer,
                         ras::image2d_depth32& depthBuffer,
                         const glm::mat4&      projMat,
                         const glm::mat4&      viewProjMat);

  void DrawFireParticles(ras::image2d_bgra&    backBuffer,
                         ras::image2d_depth32& depthBuffer,
                         const glm::mat4&      projMat,
                         const glm::mat4&      viewProjMat);

  Camera m_Camera;

  ras::raster_cache<IslandVSOutput>   m_IslandRasterCache;
  ras::raster_cache<RiverVSOutput>    m_RiverRasterCache;
  ras::raster_cache<ParticleVSOutput> m_ParticleRasterCache;

  Mesh              m_IslandMesh;
  Mesh              m_RiverMesh;
  ras::image2d_rgba m_RiverTexture;

  ParticleSystem m_SnowParticles;
  float          m_SnowParticleEmissionCounter;

  ParticleSystem m_FireParticles;
  float          m_FireParticleEmissionCounter;

  std::default_random_engine m_Random;

  double m_RiverUvTime;
  bool   m_IsCameraMouseDown;
};
