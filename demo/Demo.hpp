#pragma once

#include <SDL2/SDL.h>
#include <cppgfx.hpp>
#include <glm/vec2.hpp>

class Demo
{
public:
  Demo() = default;

  Demo(const Demo&) = default;

  auto operator=(const Demo&) -> Demo& = default;

  Demo(Demo&&) noexcept = default;

  auto operator=(Demo&&) noexcept -> Demo& = default;

  virtual ~Demo() noexcept = default;

  virtual void Update(float elapsedTime);

  virtual void Draw(ras::image2d_bgra& backBuffer, ras::image2d_depth32& depthBuffer) = 0;

  virtual void IsMouseButtonDown(int button);

  virtual void IsMouseButtonUp(int button);

  virtual void MouseHasMoved(glm::vec2 delta);

  virtual void MouseWheelHasMoved(glm::vec2 delta);
};
