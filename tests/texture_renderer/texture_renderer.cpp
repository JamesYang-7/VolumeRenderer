#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <iostream>
#include <fstream>
#include <vector>
#include "vol_renderer/common.h"
#include "vol_renderer/camera_gl.h"
#include "vol_renderer/shader.h"
#include "vol_renderer/gui.h"

// Main window
GLFWwindow* window = nullptr;

// camera
GLCamera main_camera(glm::vec3(0.0f, 0.0f, 3.0f));

std::string vertexShaderFile = "shaders/texture.vs";
std::string fragmentShaderFile = "shaders/texture.frag";

int main() {
    // Initialize GLFW
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW" << std::endl;
        return -1;
    }

    // Set OpenGL version and profile
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    // Create window
    window = glfwCreateWindow(WIDTH, HEIGHT, "Texture Renderer", nullptr, nullptr);
    if (!window) {
        std::cerr << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);

    // Initialize GLEW
    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK) {
        std::cerr << "Failed to initialize GLEW" << std::endl;
        return -1;
    }

    GUI gui(window);
    gui.init();

    // Set background color
    glClearColor(0.1f, 0.1f, 0.1f, 1.0f);

    // load texture
    GLuint texture;
    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);

    // Define the texture size and format
    GLsizei width = 800, height = 600;
    std::vector<GLubyte> pixelData(width * height * 4, 127);
    // Initialize a red line
    for (int i = 0; i < width; ++i) {
        for (int j = 0; j < 3; ++j) {
            pixelData[(j * width + i) * 4 + 0] = 255; // R
            pixelData[(j * width + i) * 4 + 1] = 0;   // G
            pixelData[(j * width + i) * 4 + 2] = 0;   // B
            pixelData[(j * width + i) * 4 + 3] = 255; // A
        }
    }
    // Upload the texture data
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, pixelData.data());
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    float vertices[] = {
        // Positions    // TexCoords
        -1.0f, -1.0f,   0.0f, 0.0f,
        1.0f, -1.0f,   1.0f, 0.0f,
        -1.0f,  1.0f,   0.0f, 1.0f,
        1.0f,  1.0f,   1.0f, 1.0f
    };

    // Generate and bind VAO and VBO
    GLuint VAO, VBO;
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);
    glBindVertexArray(VAO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (GLvoid*)0);
    glEnableVertexAttribArray(0);
    glBindVertexArray(0);

    // Compile and link shaders
    Shader shader(vertexShaderFile.c_str(), fragmentShaderFile.c_str());
    shader.use();

    int y = 0;

    Timer timer;
    // Main loop
    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();
        // per-frame time logic
        float delta_time = timer.getDeltaTime();
        gui.process_input(delta_time);
        if (delta_time < 0.05f) continue;
        timer.update();

        y = (y + 1) % height;
        int j = (y-3 + height) % height;
        for (int i = 0; i < width; ++i) {
            // move up the line
            pixelData[(y * width + i) * 4 + 0] = 255; // R
            pixelData[(y * width + i) * 4 + 1] = 0;   // G
            pixelData[(y * width + i) * 4 + 2] = 0;   // B
            pixelData[(y * width + i) * 4 + 3] = 255; // A

            pixelData[(j * width + i) * 4 + 0] = 127; // R
            pixelData[(j * width + i) * 4 + 1] = 127; // G
            pixelData[(j * width + i) * 4 + 2] = 127; // B
            pixelData[(j * width + i) * 4 + 3] = 127; // A
        }

        // Clear the color buffers
        glClear(GL_COLOR_BUFFER_BIT);

        shader.use();
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, pixelData.data());
        glBindVertexArray(VAO);
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

        // Unbind the VAO
        glBindVertexArray(0);

        // Swap buffers
        glfwSwapBuffers(window);
    }

    // Clean up
    glDeleteVertexArrays(1, &VAO);
    glDeleteBuffers(1, &VBO);
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
