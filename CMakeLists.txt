cmake_minimum_required(VERSION 3.20)

project(cppgfx LANGUAGES CXX)

option(CPPRASTER_BUILD_DEMO "Build the demo app?" ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(cppgfx INTERFACE)
target_sources(cppgfx INTERFACE include/cppgfx.hpp)

target_include_directories(cppgfx INTERFACE
  ${CMAKE_CURRENT_SOURCE_DIR}/Include
)

if (MSVC)
  target_compile_options(cppgfx INTERFACE /W4 /WX)
  target_compile_definitions(cppgfx INTERFACE "_CRT_SECURE_NO_WARNINGS")
else ()
  target_compile_options(cppgfx INTERFACE -Wall -Wextra -pedantic -Werror)
endif ()

if (CPPRASTER_BUILD_DEMO)
  add_subdirectory(Demo)
endif ()
