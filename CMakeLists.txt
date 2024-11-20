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

###############################################################################
# CUDA compiler setup
###############################################################################

# Figure out CUDA version
if(CMAKE_CUDA_COMPILER_LOADED)
	if (CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA" AND CMAKE_CUDA_COMPILER_VERSION MATCHES "^([0-9]+\\.[0-9]+)")
		set(CUDA_VERSION "${CMAKE_MATCH_1}")
	endif()
endif()

# Adapted from the CMake source code at https://github.com/Kitware/CMake/blob/master/Modules/FindCUDA/select_compute_arch.cmake
# Simplified to return a semicolon-separated list of the compute capabilities of installed devices
function(VOLRENDERER_AUTODETECT_CUDA_ARCHITECTURES OUT_VARIABLE)
	if (NOT VOLRENDERER_AUTODETECT_CUDA_ARCHITECTURES_OUTPUT)
		if (CMAKE_CUDA_COMPILER_LOADED) # CUDA as a language
			set(file "${PROJECT_BINARY_DIR}/detect_volrenderer_cuda_architectures.cu")
		else()
			set(file "${PROJECT_BINARY_DIR}/detect_volrenderer_cuda_architectures.cpp")
		endif()

		file(WRITE ${file} ""
			"#include <cuda_runtime.h>\n"
			"#include <cstdio>\n"
			"int main() {\n"
			"	int count = 0;\n"
			"	if (cudaSuccess != cudaGetDeviceCount(&count)) return -1;\n"
			"	if (count == 0) return -1;\n"
			"	for (int device = 0; device < count; ++device) {\n"
			"		cudaDeviceProp prop;\n"
			"		if (cudaSuccess == cudaGetDeviceProperties(&prop, device)) {\n"
			"			std::printf(\"%d%d\", prop.major, prop.minor);\n"
			"			if (device < count - 1) std::printf(\";\");\n"
			"		}\n"
			"	}\n"
			"	return 0;\n"
			"}\n"
		)

		if (CMAKE_CUDA_COMPILER_LOADED) # CUDA as a language
			try_run(run_result compile_result ${PROJECT_BINARY_DIR} ${file} RUN_OUTPUT_VARIABLE compute_capabilities)
		else()
			try_run(
				run_result compile_result ${PROJECT_BINARY_DIR} ${file}
				CMAKE_FLAGS "-DINCLUDE_DIRECTORIES=${CUDA_INCLUDE_DIRS}"
				LINK_LIBRARIES ${CUDA_LIBRARIES}
				RUN_OUTPUT_VARIABLE compute_capabilities
			)
		endif()

		if (run_result EQUAL 0)
			# If the user has multiple GPUs with the same compute capability installed, list that capability only once.
			list(REMOVE_DUPLICATES compute_capabilities)
			set(VOLRENDERER_AUTODETECT_CUDA_ARCHITECTURES_OUTPUT ${compute_capabilities} CACHE INTERNAL "Returned GPU architectures from detect_gpus tool" FORCE)
		endif()
	endif()

	if (NOT VOLRENDERER_AUTODETECT_CUDA_ARCHITECTURES_OUTPUT)
		message(STATUS "Automatic GPU detection failed. Building for Turing and Ampere as a best guess.")
		set(${OUT_VARIABLE} "75;86" PARENT_SCOPE)
	else()
		set(${OUT_VARIABLE} ${VOLRENDERER_AUTODETECT_CUDA_ARCHITECTURES_OUTPUT} PARENT_SCOPE)
	endif()
endfunction()

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)
set(CUDA_LINK_LIBRARIES_KEYWORD PUBLIC)

get_directory_property(VOLRENDERER_HAS_PARENT PARENT_DIRECTORY)
if (DEFINED ENV{VOLRENDERER_CUDA_ARCHITECTURES})
	message(STATUS "Obtained CUDA architectures from environment variable VOLRENDERER_CUDA_ARCHITECTURES=$ENV{VOLRENDERER_CUDA_ARCHITECTURES}")
	set(CMAKE_CUDA_ARCHITECTURES $ENV{VOLRENDERER_CUDA_ARCHITECTURES})
elseif (VOLRENDERER_CUDA_ARCHITECTURES)
	message(STATUS "Obtained CUDA architectures from CMake variable VOLRENDERER_CUDA_ARCHITECTURES=${VOLRENDERER_CUDA_ARCHITECTURES}")
	set(CMAKE_CUDA_ARCHITECTURES ${VOLRENDERER_CUDA_ARCHITECTURES})
else()
	message(STATUS "Obtained CUDA architectures automatically from installed GPUs")
	VOLRENDERER_AUTODETECT_CUDA_ARCHITECTURES(CMAKE_CUDA_ARCHITECTURES)
endif()

# If the CUDA version does not support the chosen architecture, target
# the latest supported one instead.
if (CUDA_VERSION VERSION_LESS 11.0)
	set(LATEST_SUPPORTED_CUDA_ARCHITECTURE 75)
elseif (CUDA_VERSION VERSION_LESS 11.1)
	set(LATEST_SUPPORTED_CUDA_ARCHITECTURE 80)
elseif (CUDA_VERSION VERSION_LESS 11.8)
	set(LATEST_SUPPORTED_CUDA_ARCHITECTURE 86)
else()
	set(LATEST_SUPPORTED_CUDA_ARCHITECTURE 90)
endif()

if (CUDA_VERSION VERSION_GREATER_EQUAL 12.0)
	set(EARLIEST_SUPPORTED_CUDA_ARCHITECTURE 50)
else()
	set(EARLIEST_SUPPORTED_CUDA_ARCHITECTURE 20)
endif()

foreach (CUDA_CC IN LISTS CMAKE_CUDA_ARCHITECTURES)
	if (CUDA_CC GREATER ${LATEST_SUPPORTED_CUDA_ARCHITECTURE})
		message(WARNING "CUDA version ${CUDA_VERSION} is too low for detected architecture ${CUDA_CC}. Targeting the highest supported architecture ${LATEST_SUPPORTED_CUDA_ARCHITECTURE} instead.")
		list(REMOVE_ITEM CMAKE_CUDA_ARCHITECTURES ${CUDA_CC})
		if (NOT CMAKE_CUDA_ARCHITECTURES)
			list(APPEND CMAKE_CUDA_ARCHITECTURES ${LATEST_SUPPORTED_CUDA_ARCHITECTURE})
		endif()
	endif()

	if (CUDA_CC LESS ${EARLIEST_SUPPORTED_CUDA_ARCHITECTURE})
		message(ERROR "CUDA version ${CUDA_VERSION} no longer supports detected architecture ${CUDA_CC}. Targeting the lowest supported architecture ${EARLIEST_SUPPORTED_CUDA_ARCHITECTURE} instead.")
		list(REMOVE_ITEM CMAKE_CUDA_ARCHITECTURES ${CUDA_CC})
		if (NOT CMAKE_CUDA_ARCHITECTURES)
			list(APPEND CMAKE_CUDA_ARCHITECTURES ${EARLIEST_SUPPORTED_CUDA_ARCHITECTURE})
		endif()
	endif()
endforeach(CUDA_CC)

if (NOT CMAKE_CUDA_ARCHITECTURES)
	list(APPEND CMAKE_CUDA_ARCHITECTURES ${LATEST_SUPPORTED_CUDA_ARCHITECTURE})
endif()

# Sort the list to obtain lowest architecture that must be compiled for.
list(SORT CMAKE_CUDA_ARCHITECTURES COMPARE NATURAL ORDER ASCENDING)
list(GET CMAKE_CUDA_ARCHITECTURES 0 MIN_GPU_ARCH)

string(REPLACE "-virtual" "" MIN_GPU_ARCH "${MIN_GPU_ARCH}")

message(STATUS "Targeting CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
if (VOLRENDERER_HAS_PARENT)
	set(VOLRENDERER_CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES} PARENT_SCOPE)
	set(VOLRENDERER_CUDA_VERSION ${CUDA_VERSION} PARENT_SCOPE)
endif()

if (MIN_GPU_ARCH LESS_EQUAL 70)
	message(WARNING
		"Fully fused MLPs do not support GPU architectures of 70 or less. "
		"Falling back to CUTLASS MLPs. Remove GPU architectures 70 and lower "
		"to allow maximum performance"
	)
endif()

if (CUDA_VERSION VERSION_LESS 10.2)
	message(FATAL_ERROR "CUDA version too low. tiny-cuda-nn require CUDA 10.2 or higher.")
endif()

list(APPEND VOLRENDERER_CUDA_LIBRARIES cuda)
list(APPEND VOLRENDERER_DEFINITIONS -DVOLRENDERER_MIN_GPU_ARCH=${MIN_GPU_ARCH})
if (VOLRENDERER_ALLOW_CUBLAS_CUSOLVER AND CUDA_VERSION VERSION_GREATER_EQUAL 11.0)
	# Only compile the shampoo optimizer if
	# a new enough cuBLAS version is available.
	list(APPEND VOLRENDERER_CUDA_LIBRARIES cublas)
	list(APPEND VOLRENDERER_DEFINITIONS -DVOLRENDERER_SHAMPOO)
endif()

if (VOLRENDERER_HAS_PARENT)
	set(VOLRENDERER_DEFINITIONS ${VOLRENDERER_DEFINITIONS} PARENT_SCOPE)
endif()

if (MSVC)
	list(APPEND CUDA_NVCC_FLAGS "-Xcompiler=/bigobj")
else()
	list(APPEND CUDA_NVCC_FLAGS "-Xcompiler=-Wno-float-conversion")
	list(APPEND CUDA_NVCC_FLAGS "-Xcompiler=-fno-strict-aliasing")
	list(APPEND CUDA_NVCC_FLAGS "-Xcudafe=--diag_suppress=unrecognized_gcc_pragma")
endif()
list(APPEND CUDA_NVCC_FLAGS "--extended-lambda")
list(APPEND CUDA_NVCC_FLAGS "--expt-relaxed-constexpr")

###############################################################################
# Dependencies
###############################################################################

add_subdirectory("dependencies/glew-cmake")
add_subdirectory("dependencies/glfw")

list(APPEND VOLRENDERER_INCLUDE_DIRS
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/dependencies/glew-cmake/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/dependencies/glfw/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/dependencies/glm"
    "${CMAKE_CURRENT_SOURCE_DIR}/dependencies/imgui"
	"${CMAKE_CURRENT_SOURCE_DIR}/dependencies/imgui/backends"
    ${CUDA_INCLUDE_DIRS}
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

list(APPEND VOLRENDERER_GL_LIBRARIES
    libglew_static
    glfw
)

list(APPEND VOLRENDERER_LINK_LIBRARIES
    ${VOLRENDERER_GL_LIBRARIES}
    ${VOLRENDERER_CUDA_LIBRARIES}
)

###############################################################################
# Targets
###############################################################################

add_executable(mesh_renderer src/mesh_renderer.cpp src/obj.cpp src/gui.cpp)
target_include_directories(mesh_renderer PRIVATE ${VOLRENDERER_INCLUDE_DIRS})
target_link_libraries(mesh_renderer PRIVATE ${VOLRENDERER_GL_LIBRARIES})

add_executable(device_test src/device_test.cpp)
target_include_directories(device_test PRIVATE ${VOLRENDERER_INCLUDE_DIRS})
target_link_libraries(device_test PRIVATE ${VOLRENDERER_GL_LIBRARIES})

add_executable(texture_renderer src/texture_renderer.cpp src/gui.cpp)
target_include_directories(texture_renderer PRIVATE ${VOLRENDERER_INCLUDE_DIRS})
target_link_libraries(texture_renderer PRIVATE ${VOLRENDERER_GL_LIBRARIES})

add_executable(hello_world_cuda src/hello_world_cuda.cu)
target_include_directories(hello_world_cuda PRIVATE ${VOLRENDERER_INCLUDE_DIRS})
target_link_libraries(hello_world_cuda PRIVATE ${VOLRENDERER_LINK_LIBRARIES})

add_executable(hello_gui src/hello_gui.cpp)
target_include_directories(hello_gui PRIVATE ${VOLRENDERER_INCLUDE_DIRS})
target_link_libraries(hello_gui PRIVATE ${VOLRENDERER_GL_LIBRARIES} imgui)
