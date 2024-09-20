#pragma once

#include <cppgfx.hpp>
#include <glm/vec2.hpp>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>
#include <string>
#include <vector>

struct ParticleVertex {
  glm::vec2 corner;
  glm::vec3 position;
  glm::vec3 velocity;
  glm::vec4 randomValues;
  float age{};
};

class ParticleSystem {
public:
  void update(float deltaTime);

  void emitParticle(const glm::vec3& position, const glm::vec3& velocity);

  template <typename DrawAction>
  void draw(const DrawAction& drawAction) {
    constexpr auto indices = std::array{
        uint16_t(0), uint16_t(1), uint16_t(2), uint16_t(0), uint16_t(2), uint16_t(3),
    };
    const ras::index_view indexView{indices};

    for (const auto& particle : m_particles) {
      const auto particleAge = particleLifeTime - particle.ttl;

      const auto vertices = std::array{
          ParticleVertex{
              .corner = {-1, -1},
              .position = particle.position,
              .velocity = particle.velocity,
              .randomValues = particle.randomValues,
              .age = particleAge,
          },
          ParticleVertex{
              .corner = {1, -1},
              .position = particle.position,
              .velocity = particle.velocity,
              .randomValues = particle.randomValues,
              .age = particleAge,
          },
          ParticleVertex{
              .corner = {1, 1},
              .position = particle.position,
              .velocity = particle.velocity,
              .randomValues = particle.randomValues,
              .age = particleAge,
          },
          ParticleVertex{
              .corner = {-1, 1},
              .position = particle.position,
              .velocity = particle.velocity,
              .randomValues = particle.randomValues,
              .age = particleAge,
          },
      };

      const auto vertexView = ras::vertex_view<ParticleVertex>{vertices};

      drawAction(vertexView, indexView, 0U, 6U);
    }
  }

  size_t liveParticleCount() const {
    return m_particles.size();
  }

  std::string m_name;
  glm::vec3 m_position;

  float particleLifeTime = 1.0F;
  float particleLifeTimeRandomness = 0.0F;

  glm::vec2 horizontalVelocity{0.0F, 0.0F};
  glm::vec2 verticalVelocity{0.0F, 0.0F};
  glm::vec3 gravity{0.0F, 1.0F, 0.0F};
  float endVelocity = 1.0F;
  glm::vec4 minColor{1.0F, 1.0F, 1.0F, 1.0F};
  glm::vec4 maxColor{1.0F, 1.0F, 1.0F, 1.0F};
  glm::vec2 rotateSpeed{0.0F, 0.0F};
  glm::vec2 startSize{1.0F, 1.0F};
  glm::vec2 endSize{1.0F, 1.0F};

private:
  struct Particle {
    float ttl;
    glm::vec3 position;
    glm::vec3 velocity;
    glm::vec4 randomValues;
  };

  std::vector<Particle> m_particles;
};
