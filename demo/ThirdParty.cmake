include(FetchContent)

macro(SetOption name onOrOff)
  set(${name} ${onOrOff} CACHE INTERNAL "")
endmacro()

SetOption(BUILD_TESTS OFF)
SetOption(SDL_AUDIO OFF)
SetOption(SDL_CAMERA OFF)
SetOption(SDL_DISABLE_INSTALL ON)
SetOption(SDL_DISABLE_UNINSTALL ON)
SetOption(SDL2_DISABLE_INSTALL ON)
SetOption(SDL2_DISABLE_UNINSTALL ON)
SetOption(SDL_HIDAPI OFF)
#SetOption(SDL_METAL OFF)
SetOption(SDL_VULKAN OFF)
SetOption(SDL_SHARED OFF)
SetOption(SDL_STATIC ON)
SetOption(SDL_TESTS OFF)
SetOption(TINYGLTF_BUILD_LOADER_EXAMPLE OFF)
SetOption(TINYGLTF_INSTALL OFF)

# SDL
FetchContent_Declare(
  SDL
  GIT_REPOSITORY https://github.com/libsdl-org/SDL.git
  GIT_TAG SDL2
)

FetchContent_MakeAvailable(SDL)

# glm
FetchContent_Declare(
  glm
  GIT_REPOSITORY https://github.com/g-truc/glm.git
  GIT_TAG 1.0.1
)

FetchContent_MakeAvailable(glm)

# tinygltf
FetchContent_Declare(
  tinygltf
  GIT_REPOSITORY https://github.com/syoyo/tinygltf.git
  GIT_TAG v2.9.0
)

FetchContent_MakeAvailable(tinygltf)

# stb
FetchContent_Declare(
  stb
  GIT_REPOSITORY https://github.com/nothings/stb.git
)

FetchContent_GetProperties(stb)

# CMakeRC
FetchContent_Declare(
  cmrc
  GIT_REPOSITORY https://github.com/vector-of-bool/cmrc.git
  GIT_TAG 2.0.1
)

FetchContent_MakeAvailable(cmrc)
