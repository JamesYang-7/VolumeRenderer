#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <cuda_gl_interop.h>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>
#include <tiny-cuda-nn/common_host.h>
#include <json/json.hpp>
#include "vol_renderer/camera.h"
#include "vol_renderer/ray.h"
#include "vol_renderer/bbox.h"
#include "vol_renderer/volume_data.h"
#include "vol_renderer/data_loader.h"
#include "vol_renderer/transfer_function.h"
#include "vol_renderer/timer.h"

using json = nlohmann::json;


__global__ void processTextureKernel(uchar4* frame_buffer, uint32_t width, uint32_t height) {
    uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    uint32_t idx = y * width + x;

    frame_buffer[idx].x = (uint32_t)((float)x / width * 255.0f);
    frame_buffer[idx].y = (uint32_t)((float)y / height * 255.0f);
    frame_buffer[idx].z = 127;
}

__host__ __device__ void composite(
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

__global__ void rayTracingKernel(
    uchar4* frame_buffer, uint32_t width,
    uint32_t height, const Camera* camera,
    const AABB* bbox,
    const VolumeData<float>* volume,
    TransferFunctionType tf_type,
    const float step_size = 0.01f
)
{
    uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    uint32_t idx = y * width + x;

    float u = ((float)x + 0.5f) / width;
    float v = ((float)y + 0.5f) / height;

    Ray ray = camera->generateRay(u, v);
    float tmin, tmax;
    
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
            glm::vec3 local_pos = bbox->getLocalPos(p);
            float val = volume->at(local_pos);
            glm::vec4 rgba = getColor(val, tf_type);
            color = glm::vec3(rgba.x, rgba.y, rgba.z);
            alpha = rgba.w;
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

void rayTracingHost(
    uchar4* frame_buffer, uint32_t width,
    uint32_t height, const Camera* camera,
    const AABB* bbox,
    const VolumeData<float>* volume,
    TransferFunctionType tf_type,
    const float step_size = 0.01f
)
{
    for (uint32_t x = 0; x < width; ++x) {
        for (uint32_t y = 0; y < height; ++y) {
            uint32_t idx = y * width + x;
            float u = ((float)x + 0.5f) / width;
            float v = ((float)y + 0.5f) / height;

            Ray ray = camera->generateRay(u, v);
            float tmin, tmax;
            glm::vec3 color(0.0f, 0.0f, 0.5f);
            float alpha = 0.2f;
            glm::vec3 bg_color(0.0f);
            float bg_alpha = 0.0f;
            glm::vec3 out_color(0.0f);
            float out_alpha = 0.0f;
            unsigned char r = 0, g = 0, b = 0, a = 0;
            if (bbox->ray_intersect(&ray, &tmin, &tmax)) {
                float t = tmax;
                while (t > tmin) {
                    glm::vec3 p = ray.at(t);
                    glm::vec3 local_pos = bbox->getLocalPos(p);
                    float val = volume->at(local_pos);
                    glm::vec4 rgba = getColor(val, tf_type);
                    color = glm::vec3(rgba.x, rgba.y, rgba.z);
                    alpha = rgba.w;
                    composite(color, alpha, bg_color, bg_alpha, out_color, out_alpha);
                    bg_color = out_color;
                    bg_alpha = out_alpha;
                    t -= step_size;
                }
                r = (unsigned char)(255.0f * out_color.x);
                g = (unsigned char)(255.0f * out_color.y);
                b = (unsigned char)(255.0f * out_color.z);
                a = (unsigned char)(255.0f * out_alpha);
            }
            frame_buffer[idx].x = r;
            frame_buffer[idx].y = g;
            frame_buffer[idx].z = b;
            frame_buffer[idx].w = a;
        }
    }
}


struct VolumeRenderer {
    uint32_t res_x = 0;
    uint32_t res_y = 0;
    GLuint gl_texture_id;
    cudaGraphicsResource* cuda_resource;
    uchar4* frame_buffer;
    size_t size;
    TransferFunctionType tf = TransferFunctionType::GRAYSCALE;
    float sampling_rate = 10.0f; // inverse of step size
    bool on_host = false;

    VolumeRenderer(uint32_t res_x, uint32_t res_y, bool on_host=false) : res_x(res_x), res_y(res_y), on_host(on_host) {
        size = res_x * res_y * sizeof(uchar4);
        if (on_host) {
            frame_buffer = (uchar4*)malloc(size);
            memset(frame_buffer, 255, size);
        } else {
            cudaMalloc(&frame_buffer, size);
            cudaMemset(frame_buffer, 255, size);
        }
    }

    ~VolumeRenderer() {
        if (on_host) {
            free(frame_buffer);
        } else {
            cudaGraphicsUnregisterResource(cuda_resource);
            cudaFree(frame_buffer);
        }
    }

    void init() {
        glGenTextures(1, &gl_texture_id);
        glBindTexture(GL_TEXTURE_2D, gl_texture_id);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, res_x, res_y, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        if (!on_host) cudaGraphicsGLRegisterImage(&cuda_resource, gl_texture_id, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsSurfaceLoadStore);
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

    void render(const Camera& camera, const AABB& bbox, const VolumeData<float>* volume) {
        if (on_host) {
            render_host(camera, bbox, volume);
        } else {
            render_device(camera, bbox, volume);
        }
    }

    void render_device(const Camera& camera, const AABB& bbox, const VolumeData<float>* volume) {
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
        rayTracingKernel<<<numBlocks, blockSize>>>(frame_buffer, res_x, res_y, d_camera, d_bbox, volume, tf, 1.0f / sampling_rate);
        cudaDeviceSynchronize();
        cudaMemcpyToArray(array, 0, 0, frame_buffer, size, cudaMemcpyDeviceToDevice);
        cudaGraphicsUnmapResources(1, &cuda_resource);
        cudaFree(d_camera);
        cudaFree(d_bbox);
    }

    void render_host(const Camera& camera, const AABB& bbox, const VolumeData<float>* volume) {
        rayTracingHost(frame_buffer, res_x, res_y, &camera, &bbox, volume, tf, 1.0f / sampling_rate);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, res_x, res_y, 0, GL_RGBA, GL_UNSIGNED_BYTE, frame_buffer);
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
    GLFWwindow* window = glfwCreateWindow(width, height, "Hello GUI", nullptr, nullptr);
    if (!window) {
        std::cerr << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return nullptr;
    }
    glfwMakeContextCurrent(window);
    glfwSwapInterval(0); // Disable VSync

    // Initialize GLEW
    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK) {
        std::cerr << "Failed to initialize GLEW" << std::endl;
        return nullptr;
    }
    return window;
} 

int main() {
    std::ifstream config_file("../config/config.json");
    json args = json::parse(config_file);
    uint32_t WIDTH = args["WIDTH"].get<uint32_t>();
    uint32_t HEIGHT = args["HEIGHT"].get<uint32_t>();
    uint32_t VIS_WIDTH = args["VIS_WIDTH"].get<uint32_t>();
    uint32_t CONFIG_WIDTH = WIDTH - VIS_WIDTH;
    float CAMERA_RADIUS = args["CAMERA_RADIUS"].get<float>();
    float CAMERA_SPEED = args["CAMERA_SPEED"].get<float>();
    bool USE_CPU = args["USE_CPU"].get<bool>();
    
    GLFWwindow* window = createWindow(WIDTH, HEIGHT, "Volume Renderer");
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

    VolumeRenderer renderer(VIS_WIDTH, HEIGHT, USE_CPU);
    renderer.init();

    const char* VIS_WINDOW_NAME = "Volume";
    ImVec2 VIS_WINDOW_POS = ImVec2(0, 0);
    ImVec2 VIS_WINDOW_SIZE = ImVec2(VIS_WIDTH, HEIGHT);
    const char* CONFIG_WINDOW_NAME = "Settings";
    ImVec2 CONFIG_WINDOW_POS = ImVec2(VIS_WIDTH, 0);
    ImVec2 CONFIG_WINDOW_SIZE = ImVec2(CONFIG_WIDTH, HEIGHT);

    // Camera
    glm::vec3 eye(-1.0f, 0.0f, 0.0f);
    glm::vec3 center(0.0f, 0.0f, 0.0f);
    glm::vec3 up = glm::vec3(args["CAMERA_UP"][0].get<float>(), args["CAMERA_UP"][1].get<float>(), args["CAMERA_UP"][2].get<float>());
    float phi = 0.0f;
    float theta = 0.0f;
    float radius = CAMERA_RADIUS;
    float camera_speed = CAMERA_SPEED;

    // load volume data
    glm::vec3 voxel_ratio(1.0f, 1.0f, 2.0f);
    DataLoader cthead_loader(std::string("../data/CThead.bin"), true);
    VolumeData<float>* cthead_h_volume = new VolumeData<float>(cthead_loader.getData(), cthead_loader.getSize(), voxel_ratio);
    VolumeData<float>* cthead_d_volume = cthead_h_volume->copyToDevice();
    DataLoader mrbrain_loader(std::string("../data/MRbrain.bin"), true);
    VolumeData<float>* mrbrain_h_volume = new VolumeData<float>(mrbrain_loader.getData(), mrbrain_loader.getSize(), voxel_ratio);
    VolumeData<float>* mrbrain_d_volume = mrbrain_h_volume->copyToDevice();
    
    VolumeData<float>* h_volume_array[] = {cthead_h_volume, mrbrain_h_volume};
    VolumeData<float>* d_volume_array[] = {cthead_d_volume, mrbrain_d_volume};
    int data_idx = 0;

    // set volume within [0, 1]^3
    glm::vec3 volume_shape = glm::vec3(cthead_h_volume->getShape()) * voxel_ratio;
    float max_dim = std::max(volume_shape.x, std::max(volume_shape.y, volume_shape.z));
    volume_shape /= max_dim;
    // move volume to the center
    AABB cthead_bbox(-volume_shape / 2.0f, volume_shape / 2.0f);

    volume_shape = glm::vec3(mrbrain_h_volume->getShape()) * voxel_ratio;
    max_dim = std::max(volume_shape.x, std::max(volume_shape.y, volume_shape.z));
    volume_shape /= max_dim;
    AABB mrbrain_bbox(-volume_shape / 2.0f, volume_shape / 2.0f);

    AABB* bbox_array[] = {&cthead_bbox, &mrbrain_bbox};

    const char* tf_names[] = {"Grayscale", "Skull", "Skin", "Nerve"};
    const char* data_names[] = {"CThead", "MRbrain"};

    // Main loop
    Timer timer;
    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();

        // Calculate delta time
        float deltaTime = timer.getDeltaTime();
        if (deltaTime > 0.033f) {
            // Update the camera
            theta = theta + camera_speed * deltaTime;
            theta = theta > 360.0f ? theta - 360.0f : theta;
            if (data_idx == 0) {
                eye = radius * glm::vec3(-cos(glm::radians(phi)) * cos(glm::radians(theta)),
                            cos(glm::radians(phi)) * sin(glm::radians(theta)),
                            sin(glm::radians(phi))
                            );
                up = glm::vec3(0.0f, 0.0f, 1.0f);
            } else {
                eye = radius * glm::vec3(cos(glm::radians(phi)) * cos(glm::radians(theta)),
                                sin(glm::radians(phi)),
                                cos(glm::radians(phi)) * sin(glm::radians(theta))
                                );
                up = glm::vec3(0.0f, 1.0f, 0.0f);
            }
            timer.update();
        }
        
        Camera camera(eye, center - eye, up, YAW, PITCH, FOV, float(VIS_WIDTH) / HEIGHT);
        if (renderer.on_host) {
            renderer.render_host(camera, *bbox_array[data_idx], h_volume_array[data_idx]);
        } else {
            renderer.render_device(camera, *bbox_array[data_idx], d_volume_array[data_idx]);
        }

        // Start the Dear ImGui frame
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        ImGui::SetWindowPos(VIS_WINDOW_NAME, VIS_WINDOW_POS);
        ImGui::SetWindowSize(VIS_WINDOW_NAME, VIS_WINDOW_SIZE);
        ImGui::Begin(VIS_WINDOW_NAME);
        ImGui::Image(renderer.gl_texture_id, VIS_WINDOW_SIZE);
        ImGui::End();

        ImGui::SetWindowPos(CONFIG_WINDOW_NAME, CONFIG_WINDOW_POS);
        ImGui::SetWindowSize(CONFIG_WINDOW_NAME, CONFIG_WINDOW_SIZE);
        ImGui::Begin(CONFIG_WINDOW_NAME);
        ImGui::Text("%.2f FPS", io.Framerate);
        ImGui::Combo("Data", &data_idx, data_names, IM_ARRAYSIZE(data_names));
        ImGui::Combo("Transfer Function", (int*)&renderer.tf, tf_names, IM_ARRAYSIZE(tf_names));
        ImGui::SliderFloat("Sampling Rate", &renderer.sampling_rate, 1.0f, 500.0f);
        ImGui::End();

        ImGui::Render();

        glClear(GL_COLOR_BUFFER_BIT);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        
        // Swap buffers
        glfwSwapBuffers(window);
    }

    // Clean up
    cthead_h_volume->release();
    cudaFree(cthead_d_volume);
    mrbrain_h_volume->release();
    cudaFree(mrbrain_d_volume);

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
