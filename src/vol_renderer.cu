#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <cuda_gl_interop.h>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>
#include "vol_renderer/common.h"
#include "vol_renderer/shader.h"


__global__ void processTextureKernel(uchar4* frame_buffer, uint32_t width, uint32_t height) {
    uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    uint32_t idx = y * width + x;

    frame_buffer[idx].x = (uint32_t)((float)x / width * 255.0f);
    frame_buffer[idx].y = (uint32_t)((float)y / height * 255.0f);
    frame_buffer[idx].z = 127;
}

struct VolumeRenderer {
    uint32_t res_x = 0;
    uint32_t res_y = 0;
    GLuint gl_texture_id;
    cudaGraphicsResource* cuda_resource;
    cudaSurfaceObject_t surface_obj;
    uchar4* frame_buffer;
    size_t size;

    VolumeRenderer(uint32_t res_x, uint32_t res_y) : res_x(res_x), res_y(res_y) {
        size = res_x * res_y * sizeof(uchar4);
        cudaMalloc(&frame_buffer, size);
        cudaMemset(frame_buffer, 255, size);
    }

    ~VolumeRenderer() {
        cudaFree(frame_buffer);
    }

    void init() {
        glGenTextures(1, &gl_texture_id);
        glBindTexture(GL_TEXTURE_2D, gl_texture_id);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, res_x, res_y, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        cudaGraphicsGLRegisterImage(&cuda_resource, gl_texture_id, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsSurfaceLoadStore);
    }

    void render() {
        cudaArray* array;
        cudaGraphicsMapResources(1, &cuda_resource);
        cudaGraphicsSubResourceGetMappedArray(&array, cuda_resource, 0, 0);

        dim3 blockSize(16, 16);
        dim3 numBlocks((res_x + blockSize.x - 1) / blockSize.x, 
                       (res_y + blockSize.y - 1) / blockSize.y);
        processTextureKernel<<<numBlocks, blockSize>>>(frame_buffer, res_x, res_y);

        cudaMemcpyToArray(array, 0, 0, frame_buffer, size, cudaMemcpyDeviceToDevice);
        cudaDeviceSynchronize();
        cudaGraphicsUnmapResources(1, &cuda_resource);
    }
};

GLFWwindow* createWindow(int width, int height, const char* title) {
    // Initialize GLFW
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW" << std::endl;
        return nullptr;
    }

    // Set OpenGL version and profile
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    // Create window
    GLFWwindow* window = glfwCreateWindow(WIDTH, HEIGHT, "Hello GUI", nullptr, nullptr);
    if (!window) {
        std::cerr << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return nullptr;
    }
    glfwMakeContextCurrent(window);

    // Initialize GLEW
    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK) {
        std::cerr << "Failed to initialize GLEW" << std::endl;
        return nullptr;
    }
    return window;
} 

int main() {
    
    GLFWwindow*  window = createWindow(WIDTH, HEIGHT, "Volume Renderer");
    if (window == nullptr) {
        return -1;
    }
    // Set background color
    glClearColor(0.1f, 0.1f, 0.1f, 1.0f);

    // Setup Dear ImGui context
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;
    ImGui::StyleColorsDark();
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 130");

    VolumeRenderer renderer(WIDTH, HEIGHT);
    renderer.init();

    const char* VIS_WINDOW_NAME = "Volume";
    ImVec2 VIS_WINDOW_POS = ImVec2(0, 0);
    ImVec2 VIS_WINDOW_SIZE = ImVec2(WIDTH, HEIGHT);

    // Main loop
    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();

        renderer.render();

        // Start the Dear ImGui frame
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        // glBindTexture(GL_TEXTURE_2D, texture);
        ImGui::SetWindowPos(VIS_WINDOW_NAME, VIS_WINDOW_POS);
        ImGui::SetWindowSize(VIS_WINDOW_NAME, VIS_WINDOW_SIZE);
        ImGui::Begin(VIS_WINDOW_NAME);
        ImGui::Image(renderer.gl_texture_id, ImVec2(WIDTH, HEIGHT));
        ImGui::End();
        ImGui::Render();

        glClear(GL_COLOR_BUFFER_BIT);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        
        // Swap buffers
        glfwSwapBuffers(window);
    }

    // Clean up
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    // cudaGraphicsUnregisterResource(renderer.cuda_resource);
    // glDeleteTextures(1, &renderer.gl_texture_id);

    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
