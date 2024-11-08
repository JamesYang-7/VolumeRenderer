cmake_minimum_required(VERSION 3.23)

project(
	volume_renderer
	VERSION 0.1
	DESCRIPTION "A volume renderer"
	LANGUAGES CXX
)

if(MSVC)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /D_CRT_SECURE_NO_WARNINGS")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MP")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_subdirectory("dependencies/glew-cmake")
add_subdirectory("dependencies/glfw")

list(APPEND VOLRENDERER_INCLUDE_DIRS
    "${CMAKE_CURRENT_SOURCE_DIR}/include",
    "${CMAKE_CURRENT_SOURCE_DIR}/dependencies/glew-cmake/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/dependencies/glfw/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/dependencies/glm",
    "${CMAKE_CURRENT_SOURCE_DIR}/dependencies/imgui"
)

list(APPEND VOLRENDERER_LINK_LIBRARIES
    libglew_static
    glfw
)

set(VOLRENDERER_SOURCES
    "src/main.cpp"
    "src/obj.cpp"
)

add_executable(vol_renderer ${VOLRENDERER_SOURCES})
target_include_directories(vol_renderer PRIVATE ${VOLRENDERER_INCLUDE_DIRS})
target_link_libraries(vol_renderer PRIVATE ${VOLRENDERER_LINK_LIBRARIES})