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
#include "vol_renderer/camera.h"
#include "vol_renderer/ray.h"
#include "vol_renderer/bbox.h"
#include <tiny-cuda-nn/common_host.h>


__global__ void processTextureKernel(uchar4* frame_buffer, uint32_t width, uint32_t height) {
    uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    uint32_t idx = y * width + x;

    frame_buffer[idx].x = (uint32_t)((float)x / width * 255.0f);
    frame_buffer[idx].y = (uint32_t)((float)y / height * 255.0f);
    frame_buffer[idx].z = 127;
}

__device__ void composite(
    const glm::vec3& color,
    const float& alpha,
    const glm::vec3& bg_color,
    const float& bg_alpha,
    glm::vec3& out_color,
    float& out_alpha
)
{
    out_color = alpha * color + (1.0f - alpha) * bg_color;
    out_alpha = alpha + (1.0f - alpha) * bg_alpha;
}

__global__ void rayTracingKernel(uchar4* frame_buffer, uint32_t width, uint32_t height, const Camera* camera, const AABB* bbox) {
    uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    uint32_t idx = y * width + x;

    float u = ((float)x + 0.5f) / width;
    float v = ((float)y + 0.5f) / height;

    Ray ray = camera->generateRay(u, v);
    float tmin, tmax;
    float step_size = 0.1f;
    
    glm::vec3 color(0.0f, 0.0f, 0.5f);
    float alpha = 0.2f;
    glm::vec3 bg_color(0.0f);
    float bg_alpha = 0.0f;
    glm::vec3 out_color(0.0f);
    float out_alpha = 0.0f;
    unsigned char r = 0, g = 0, b = 0;
    if (bbox->ray_intersect(&ray, &tmin, &tmax)) {
        float t = tmax;
        while (t > tmin) {
            glm::vec3 p = ray.at(t);
            composite(color, alpha, bg_color, bg_alpha, out_color, out_alpha);
            bg_color = out_color;
            bg_alpha = out_alpha;
            t -= step_size;
        }
        r = (unsigned char)(255.0f * out_color.x);
        g = (unsigned char)(255.0f * out_color.y);
        b = (unsigned char)(255.0f * out_color.z);
    }
    frame_buffer[idx].x = r;
    frame_buffer[idx].y = g;
    frame_buffer[idx].z = b;
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
        cudaGraphicsUnregisterResource(cuda_resource);
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

    void renderTexture() {
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

    void render(const Camera& camera, const AABB& bbox) {
        cudaArray* array;
        cudaGraphicsMapResources(1, &cuda_resource);
        cudaGraphicsSubResourceGetMappedArray(&array, cuda_resource, 0, 0);

        dim3 blockSize(16, 16);
        dim3 numBlocks((res_x + blockSize.x - 1) / blockSize.x, 
                       (res_y + blockSize.y - 1) / blockSize.y);
        Camera* d_camera;
        AABB* d_bbox;
        cudaMalloc(&d_camera, sizeof(Camera));
        cudaMalloc(&d_bbox, sizeof(AABB));
        cudaMemcpy(d_camera, &camera, sizeof(Camera), cudaMemcpyHostToDevice);
        cudaMemcpy(d_bbox, &bbox, sizeof(AABB), cudaMemcpyHostToDevice);
        rayTracingKernel<<<numBlocks, blockSize>>>(frame_buffer, res_x, res_y, d_camera, d_bbox);
        cudaDeviceSynchronize();
        cudaMemcpyToArray(array, 0, 0, frame_buffer, size, cudaMemcpyDeviceToDevice);
        cudaGraphicsUnmapResources(1, &cuda_resource);
        cudaFree(d_camera);
        cudaFree(d_bbox);
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

    // scene
    AABB bbox(glm::vec3(-1.0f), glm::vec3(1.0f));
    glm::vec3 eye(2.0f, 2.0f, -2.0f);
    glm::vec3 center(0.0f, 0.0f, 0.0f);
    glm::vec3 up(0.0f, 0.0f, 1.0f);
    Camera camera(eye, center - eye, up);

    // Main loop
    float currentTime = 0.0f;
    float deltaTime = 0.0f;
    float lastTime = 0.0f;
    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();

        // Calculate delta time
        currentTime = glfwGetTime();
        deltaTime = currentTime - lastTime;
        if (deltaTime < 0.5f) continue;
        lastTime = currentTime;

        renderer.render(camera, bbox);

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

    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
