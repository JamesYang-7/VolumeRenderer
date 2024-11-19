// Include OpenGL headers
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <iostream>
#include <cstdio>

const GLuint WIDTH = 800, HEIGHT = 600;
GLFWwindow* window = nullptr;

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
    window = glfwCreateWindow(WIDTH, HEIGHT, "Device", nullptr, nullptr);
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

    const GLubyte* renderer = glGetString(GL_RENDERER); // Graphics card
    const GLubyte* vendor = glGetString(GL_VENDOR);     // Vendor name
    const GLubyte* version = glGetString(GL_VERSION);   // OpenGL version
    printf("Renderer: %s\n", renderer);
    printf("Vendor: %s\n", vendor);
    printf("OpenGL version supported %s\n", version);

    return 0;
}
