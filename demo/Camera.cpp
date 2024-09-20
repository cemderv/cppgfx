#include "Camera.hpp"

Camera::Camera()
    : Target(0, 0, 0)
    , ZoomDistance(15.0F)
    , Pitch(glm::radians(-5.0F))
    , Yaw(0.0F)
    , RotationSensitivity(0.005F)
    , FovDegrees(10.0F)
    , NearPlane(1.0F)
    , FarPlane(25.0F)
{
}

auto Camera::GetViewMatrix() const -> glm::mat4
{
  const glm::vec3 pos            = GetPosition();
  const glm::mat4 rotationMatrix = GetRotationMatrix();
  const glm::vec3 forward        = glm::normalize(glm::vec3{rotationMatrix[2]});

  return glm::lookAt(pos, pos + forward, glm::vec3{0.0F, 1.0F, 0.0F});
}

auto Camera::GetProjectionMatrix(int viewWidth, int viewHeight) const -> glm::mat4
{
  return glm::perspectiveFov(glm::radians(FovDegrees), float(viewWidth), float(viewHeight), NearPlane, FarPlane);
}

auto Camera::GetPosition() const -> glm::vec3
{
  const auto rotationMat = this->GetRotationMatrix();
  const auto fwd         = (rotationMat * glm::vec4{0.0F, 0.0F, -1.0F, 0.0F}).rgb();
  return Target + (fwd * ZoomDistance);
}

auto Camera::GetRotationMatrix() const -> glm::mat4
{
  const glm::mat4 identity{1.0F};

  return glm::rotate(identity, Yaw, glm::vec3{0.0F, 1.0F, 0.0F}) *
         glm::rotate(identity, Pitch, glm::vec3{1.0F, 0.0F, 0.0F});
}
