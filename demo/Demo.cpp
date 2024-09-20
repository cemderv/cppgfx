#include "Demo.hpp"

#include "Camera.hpp"

void Demo::Update(float elapsedTime)
{
  std::ignore = elapsedTime;
}

void Demo::IsMouseButtonDown(int button)
{
  std::ignore = button;
}

void Demo::IsMouseButtonUp(int button)
{
  std::ignore = button;
}

void Demo::MouseHasMoved(glm::vec2 delta)
{
  std::ignore = delta;
}

void Demo::MouseWheelHasMoved(glm::vec2 delta)
{
  std::ignore = delta;
}
