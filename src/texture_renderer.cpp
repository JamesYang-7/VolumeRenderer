#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <iostream>
#include <fstream>
#include <vector>
#include "common.h"
#include "obj.h"
#include "camera.h"
#include "shader.h"

// Window dimensions
const GLuint WIDTH = 800, HEIGHT = 600;

// Function prototypes
void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void mouse_callback(GLFWwindow* window, double xpos, double ypos);
void scroll_callback(GLFWwindow* window, double xoffset, double yoffset);
void processInput(GLFWwindow *window);

// Main window
GLFWwindow* window = nullptr;

// camera
Camera camera(glm::vec3(0.0f, 0.0f, 3.0f));
float lastX = WIDTH / 2.0f;
float lastY = HEIGHT / 2.0f;
bool firstMouse = true;

// timing
float deltaTime = 0.0f;	// time between current frame and last frame
float lastTime = 0.0f;

std::string vertexShaderFile = "../shaders/texture.vs";
std::string fragmentShaderFile = "../shaders/texture.frag";

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
    window = glfwCreateWindow(WIDTH, HEIGHT, "Mesh Viewer", nullptr, nullptr);
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

    // Set callbacks
    glfwMakeContextCurrent(window);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
    glfwSetCursorPosCallback(window, mouse_callback);
    glfwSetScrollCallback(window, scroll_callback);

    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

    // Initialize OpenGL state and mesh data
    // Set background color
    glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
    glEnable(GL_DEPTH_TEST);

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

    // Main loop
    while (!glfwWindowShouldClose(window)) {

        // per-frame time logic
        processInput(window);
        float currentTime = static_cast<float>(glfwGetTime());
        deltaTime = currentTime - lastTime;
        if (deltaTime < 0.05f) continue;
        lastTime = currentTime;

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

        // Clear the color and depth buffers
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        shader.use();
        glBindTexture(GL_TEXTURE_2D, texture);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, pixelData.data());
        glBindVertexArray(VAO);
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

        // Unbind the VAO
        glBindVertexArray(0);

        // Swap buffers
        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    // Clean up
    glDeleteVertexArrays(1, &VAO);
    glDeleteBuffers(1, &VBO);
    glfwTerminate();
    return 0;
}

void framebuffer_size_callback(GLFWwindow* window, int width, int height) {
    glViewport(0, 0, width, height);
}

// process all input: query GLFW whether relevant keys are pressed/released this frame and react accordingly
void processInput(GLFWwindow *window)
{
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);

    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
        camera.ProcessKeyboard(FORWARD, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
        camera.ProcessKeyboard(BACKWARD, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
        camera.ProcessKeyboard(LEFT, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
        camera.ProcessKeyboard(RIGHT, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS)
        camera.ProcessKeyboard(UP, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS)
        camera.ProcessKeyboard(DOWN, deltaTime);
}


// glfw: whenever the mouse moves, this callback is called
void mouse_callback(GLFWwindow* window, double xposIn, double yposIn)
{
    float xpos = static_cast<float>(xposIn);
    float ypos = static_cast<float>(yposIn);

    if (firstMouse)
    {
        lastX = xpos;
        lastY = ypos;
        firstMouse = false;
    }

    float xoffset = xpos - lastX;
    float yoffset = lastY - ypos; // reversed since y-coordinates go from bottom to top

    lastX = xpos;
    lastY = ypos;

    camera.ProcessMouseMovement(xoffset, yoffset);
}

// glfw: whenever the mouse scroll wheel scrolls, this callback is called
void scroll_callback(GLFWwindow* window, double xoffset, double yoffset)
{
    camera.ProcessMouseScroll(static_cast<float>(yoffset));
}