#include "Demo.hpp"
#include "IslandDemo.hpp"
#include <SDL2/SDL.h>
#include <iostream>
#include <memory>

#if __EMSCRIPTEN__
#include <emscripten.h>
#endif

constexpr int              WindowWidth  = 640;
constexpr int              WindowHeight = 480;
constexpr std::string_view DemoTitle    = "cppgfx Demo";

struct AppState
{
  SDL_Window*           Window{};
  SDL_Renderer*         Renderer{};
  SDL_Surface*          Surface{};
  bool                  IsRunning{};
  double                PerformanceFrequency{};
  double                PreviousTime{};
  double                FrameCountTime{};
  uint32_t              FrameCount{};
  ras::image2d_bgra     BackBuffer{};
  ras::image2d_depth32  DepthBuffer{};
  std::unique_ptr<Demo> Demo{};
};

static void ResizeRenderTargets(AppState& appState, uint32_t viewWidth, uint32_t viewHeight)
{
  if (appState.Surface != nullptr)
  {
    SDL_FreeSurface(appState.Surface);
  }

  std::cout << "(Re)creating surface with size: " << viewWidth << "x" << viewHeight << std::endl;

  appState.Surface = SDL_CreateRGBSurface(0, int(viewWidth), int(viewHeight), 32, 0, 0, 0, 0);

  std::cout << "  - creation successful" << std::endl;

  appState.BackBuffer = ras::image2d_bgra{viewWidth, viewHeight, static_cast<ras::b8g8r8a8*>(appState.Surface->pixels)};
  appState.DepthBuffer = ras::image2d_depth32{viewWidth, uint32_t(viewHeight)};
}

static void Tick(AppState& appState)
{
  SDL_Event event{};
  while (SDL_PollEvent(&event) != 0)
  {
    if (event.type == SDL_QUIT)
    {
      appState.IsRunning = false;
#if __EMSCRIPTEN__
      emscripten_cancel_main_loop();
#endif
    }
    else if (event.type == SDL_MOUSEBUTTONDOWN)
    {
      appState.Demo->IsMouseButtonDown(event.button.button);
    }
    else if (event.type == SDL_MOUSEBUTTONUP)
    {
      appState.Demo->IsMouseButtonUp(event.button.button);
    }
    else if (event.type == SDL_MOUSEMOTION)
    {
      appState.Demo->MouseHasMoved(
          glm::vec2(static_cast<float>(event.motion.xrel), static_cast<float>(event.motion.yrel)));
    }
    else if (event.type == SDL_MOUSEWHEEL)
    {
      appState.Demo->MouseWheelHasMoved(
          glm::vec2(static_cast<float>(event.wheel.x), static_cast<float>(event.wheel.y)));
    }
    else if (event.type == SDL_WINDOWEVENT)
    {
      if (event.window.event == SDL_WINDOWEVENT_RESIZED)
      {
        const auto viewWidth  = uint32_t(event.window.data1);
        const auto viewHeight = uint32_t(event.window.data2);
        ResizeRenderTargets(appState, viewWidth, viewHeight);
      }
    }
  }

  if (appState.Surface == nullptr)
  {
    int width  = 0;
    int height = 0;
    SDL_GetWindowSize(appState.Window, &width, &height);
    ResizeRenderTargets(appState, width, height);
  }

  const auto now = double(SDL_GetPerformanceCounter());
  const auto elapsedTime =
      appState.PreviousTime == 0 ? 0.0 : double((now - appState.PreviousTime) / appState.PerformanceFrequency);

  appState.PreviousTime = now;
  appState.FrameCountTime += elapsedTime;

  if (appState.FrameCountTime >= 1.0)
  {
    auto newWindowTitle = std::string(DemoTitle);
    newWindowTitle += " - ";
    newWindowTitle += std::to_string(appState.FrameCount);
    newWindowTitle += " fps";
    SDL_SetWindowTitle(appState.Window, newWindowTitle.c_str());

    appState.FrameCount     = 0;
    appState.FrameCountTime = 0.0;
  }
  else
  {
    ++appState.FrameCount;
  }

  appState.Demo->Update(float(elapsedTime));

  // Draw
  if (SDL_MUSTLOCK(appState.Surface))
  {
    SDL_LockSurface(appState.Surface);
  }

  appState.Demo->Draw(appState.BackBuffer, appState.DepthBuffer);

  if (SDL_MUSTLOCK(appState.Surface))
  {
    SDL_UnlockSurface(appState.Surface);
  }

  SDL_Texture* tex = SDL_CreateTextureFromSurface(appState.Renderer, appState.Surface);
  SDL_SetTextureScaleMode(tex, SDL_ScaleModeNearest);
  SDL_SetTextureBlendMode(tex, SDL_BLENDMODE_NONE);
  SDL_RenderCopy(appState.Renderer, tex, nullptr, nullptr);

  SDL_RenderPresent(appState.Renderer);
  SDL_DestroyTexture(tex);
}

#if __EMSCRIPTEN__
void HandleEventsEmscripten(void* arg)
{
  auto& appState = *static_cast<AppState*>(arg);
  Tick(appState);
}
#endif

auto main(int argc, char* argv[]) -> int
{
  (void)argc;
  (void)argv;

  std::cout << "Initializing Video" << std::endl;

  SDL_Init(SDL_INIT_VIDEO);

  std::cout << "Creating window and renderer" << std::endl;

  int windowFlags = 0;
#ifndef __EMSCRIPTEN__
  windowFlags |= SDL_WINDOW_RESIZABLE;
#endif
  windowFlags |= SDL_WINDOW_ALLOW_HIGHDPI;

  SDL_Window* window =
      SDL_CreateWindow("", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, WindowWidth, WindowHeight, windowFlags);

  SDL_Renderer* renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_SOFTWARE);

  std::cout << "  - creation successful" << std::endl;

  SDL_SetWindowTitle(window, DemoTitle.data());

  AppState appState{
      .Window               = window,
      .Renderer             = renderer,
      .Surface              = nullptr,
      .IsRunning            = true,
      .PerformanceFrequency = double(SDL_GetPerformanceFrequency()),
      .Demo                 = std::make_unique<IslandDemo>(),
  };

#ifdef __EMSCRIPTEN__
  emscripten_set_main_loop_arg(HandleEventsEmscripten, &appState, -1, 1);
#else
  std::cout << "Running normal main loop" << std::endl;
  while (appState.IsRunning)
  {
    Tick(appState);
  }
#endif

  std::cout << "Freeing up resources" << std::endl;

  SDL_FreeSurface(appState.Surface);
  SDL_DestroyRenderer(renderer);
  SDL_DestroyWindow(window);

  SDL_Quit();

  return 0;
}
