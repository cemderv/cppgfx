add_executable(Demo
  Camera.cpp
  Camera.hpp
  Content.cpp
  Content.hpp
  Demo.cpp
  Demo.hpp
  IslandDemo.cpp
  IslandDemo.hpp
  Main.cpp
  Mesh.hpp
  ParticleSystem.cpp
  ParticleSystem.hpp
)

find_package(OpenMP)
if (OpenMP_CXX_FOUND)
  target_link_libraries(Demo PUBLIC OpenMP::OpenMP_CXX)
endif ()

include(ThirdParty.cmake)

cmrc_add_resource_library(DemoResources
  NAMESPACE Demo
  WHENCE Resources
  PREFIX Assets
  Resources/Bunny.glb
  Resources/Island.glb
  Resources/River.glb
  Resources/River.png
)

target_link_libraries(Demo PRIVATE
  cppgfx
  glm::glm
  tinygltf
  DemoResources
  $<TARGET_NAME_IF_EXISTS:SDL2::SDL2main>
  $<IF:$<TARGET_EXISTS:SDL2::SDL2>,SDL2::SDL2,SDL2::SDL2-static>
)

target_include_directories(Demo PRIVATE
  ${stb_SOURCE_DIR}
)

target_compile_definitions(Demo PRIVATE
  -DTINYGLTF_NO_STB_IMAGE
  -DTINYGLTF_NO_INCLUDE_STB_IMAGE
  -DTINYGLTF_NO_STB_IMAGE_WRITE
  -DTINYGLTF_NO_INCLUDE_STB_IMAGE_WRITE
  -DGLM_FORCE_SWIZZLE
)

if (EMSCRIPTEN)
  if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(LinkFlags
      "SHELL:-s ASSERTIONS"
      "SHELL:-s NO_DISABLE_EXCEPTION_CATCHING"
    )
    target_link_options(Demo PRIVATE ${LinkFlags})
  endif ()
endif ()
