#include "ParticleSystem.hpp"
#include <numbers>
#include <random>

static std::default_random_engine randomEngine;

void ParticleSystem::update(float deltaTime) {
  for (Particle& particle : m_particles) {
    particle.ttl -= deltaTime;
  }

  const auto it = std::remove_if(m_particles.begin(), m_particles.end(), [](const auto& particle) {
    return particle.ttl <= 0.0F;
  });

  m_particles.erase(it, m_particles.end());
}


void ParticleSystem::emitParticle(const glm::vec3& position, const glm::vec3& velocity) {
  const auto nextRandomNumber = [] {
    return std::uniform_real_distribution<float>{0.0F, 1.0F}(randomEngine);
  };

  const float hvel = std::lerp(horizontalVelocity.x, horizontalVelocity.y, nextRandomNumber());
  const float horizontalAngle = nextRandomNumber() * std::numbers::pi_v<float>;

  glm::vec3 newVelocity{velocity};
  newVelocity.x += hvel * std::cos(horizontalAngle);
  newVelocity.z += hvel * std::sin(horizontalAngle);
  newVelocity.y += std::lerp(verticalVelocity.x, verticalVelocity.y, nextRandomNumber());

  const glm::vec4 randomValues{nextRandomNumber(), nextRandomNumber(), nextRandomNumber(), nextRandomNumber()};

  m_particles.push_back(Particle{
      .ttl = particleLifeTime,
      .position = position,
      .velocity = newVelocity,
      .randomValues = randomValues,
  });
}
