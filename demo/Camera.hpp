#pragma once

#include <cppgfx.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/mat4x4.hpp>
#include <glm/vec3.hpp>

class AppTime;

class Camera
{
public:
  Camera();

  auto GetViewMatrix() const -> glm::mat4;

  auto GetProjectionMatrix(int viewWidth, int viewHeight) const -> glm::mat4;

  auto GetPosition() const -> glm::vec3;

  glm::vec3               Target;
  float                   ZoomDistance;
  float                   Pitch;
  float                   Yaw;
  std::pair<float, float> PitchRange;
  std::pair<float, float> ZoomRange;
  float                   RotationSensitivity;
  float                   FovDegrees;
  float                   NearPlane;
  float                   FarPlane;

private:
  auto GetRotationMatrix() const -> glm::mat4;
};
