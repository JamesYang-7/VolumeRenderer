cmake_minimum_required(VERSION 3.23)

project(
	volume_renderer
	VERSION 0.1
	DESCRIPTION "A volume renderer"
	LANGUAGES CXX CUDA
)

if(MSVC)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /D_CRT_SECURE_NO_WARNINGS")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MP")
endif()

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --disable-warnings --expt-relaxed-constexpr")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_RUNTIME_LIBRARY Shared)
set(CUDA_LINK_LIBRARIES_KEYWORD PUBLIC)

if (MSVC)
	list(APPEND CUDA_NVCC_FLAGS "-Xcompiler=/bigobj")
else()
	list(APPEND CUDA_NVCC_FLAGS "-Xcompiler=-fms-extensions")
endif()
list(APPEND CUDA_NVCC_FLAGS "--extended-lambda")
list(APPEND CUDA_NVCC_FLAGS "--expt-relaxed-constexpr")

###############################################################################
# Dependencies
###############################################################################

# tcnn
set(TCNN_BUILD_BENCHMARK OFF)
set(TCNN_BUILD_EXAMPLES OFF)
add_subdirectory(dependencies/tiny-cuda-nn)
set(CMAKE_CUDA_ARCHITECTURES ${TCNN_CUDA_ARCHITECTURES})
get_target_property(TCNN_INCLUDE_DIRECTORIES tiny-cuda-nn INCLUDE_DIRECTORIES)
list(APPEND VOLRENDERER_LINK_LIBRARIES tiny-cuda-nn)

# glew, glfw
add_subdirectory("dependencies/glew-cmake")
add_subdirectory("dependencies/glfw")

list(APPEND VOLRENDERER_INCLUDE_DIRS
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/dependencies/glew-cmake/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/dependencies/glfw/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/dependencies/glm"
    "${CMAKE_CURRENT_SOURCE_DIR}/dependencies/imgui"
	"${CMAKE_CURRENT_SOURCE_DIR}/dependencies/imgui/backends"
    ${TCNN_INCLUDE_DIRECTORIES}
)

# imgui
list(APPEND IMGUI_SOURCES
	"${CMAKE_CURRENT_SOURCE_DIR}/dependencies/imgui/imgui.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/dependencies/imgui/backends/imgui_impl_glfw.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/dependencies/imgui/backends/imgui_impl_opengl3.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/dependencies/imgui/imgui_draw.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/dependencies/imgui/imgui_tables.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/dependencies/imgui/imgui_widgets.cpp"
)
list(APPEND IMGUI_INCLUDE_DIRS
	"${CMAKE_CURRENT_SOURCE_DIR}/dependencies/imgui"
	"${CMAKE_CURRENT_SOURCE_DIR}/dependencies/imgui/backends"
)
add_library(imgui ${IMGUI_SOURCES})
target_include_directories(imgui PRIVATE ${IMGUI_INCLUDE_DIRS})
target_link_libraries(imgui PRIVATE glfw)

list(APPEND VOLRENDERER_LINK_LIBRARIES
	libglew_static
	glfw
	imgui
	tiny-cuda-nn
)

list(APPEND VOLRENDERER_SOURCES
	src/vol_renderer.cu
	src/camera.cu
	src/bbox.cu
	src/data_loader.cpp
	src/volume_data.cu
)

###############################################################################
# Targets
###############################################################################

add_executable(vol_renderer ${VOLRENDERER_SOURCES})
set_target_properties(vol_renderer PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_include_directories(vol_renderer PRIVATE ${VOLRENDERER_INCLUDE_DIRS})
target_link_libraries(vol_renderer PRIVATE ${VOLRENDERER_LINK_LIBRARIES})

add_subdirectory(tests)
