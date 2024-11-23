#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <iostream>
#include <fstream>
#include <vector>
#include "vol_renderer/common.h"
#include "vol_renderer/obj.h"
#include "vol_renderer/camera_gl.h"
#include "vol_renderer/shader.h"
#include "vol_renderer/gui.h"

// Main window
GLFWwindow* window = nullptr;
// camera
GLCamera main_camera(glm::vec3(0.0f, 0.0f, 3.0f));

std::string vertexShaderFile = "shaders/mesh.vs";
std::string fragmentShaderFile = "shaders/mesh.frag";

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
    window = glfwCreateWindow(WIDTH, HEIGHT, "Mesh Renderer", nullptr, nullptr);
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
    glfwSetWindowUserPointer(window, &main_camera);
    GUI gui(window);
    gui.init();

    // Initialize OpenGL state and mesh data
    // Set background color
    glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
    // load mesh
    std::vector<glm::vec3> vertices;
    std::vector<GLuint> faces;
    readOBJ("data/cube.obj", vertices, faces);
    // Compile and link shaders
    Shader shader(vertexShaderFile.c_str(), fragmentShaderFile.c_str());

    // Generate and bind VAO and VBO
    GLuint VAO, VBO, EBO;
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);
    glGenBuffers(1, &EBO);

    glBindVertexArray(VAO);

    // Load vertices into VBO
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(glm::vec3), vertices.data(), GL_STATIC_DRAW);

    // Load indices into EBO
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, faces.size() * sizeof(GLuint), faces.data(), GL_STATIC_DRAW);

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), (GLvoid*)0);
    glEnableVertexAttribArray(0);

    glBindVertexArray(0);

    // Use the shader program
    shader.use();

    // Main loop
    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();
        // per-frame time logic
        gui.process_input();

        // Clear the color buffers
        glClear(GL_COLOR_BUFFER_BIT);

        shader.use();

        glm::mat4 view = main_camera.GetViewMatrix();
        shader.setMat4("view", view);
        glm::mat4 projection = glm::perspective(glm::radians(main_camera.Zoom), (GLfloat)WIDTH / (GLfloat)HEIGHT, 0.1f, 100.0f);
        shader.setMat4("projection", projection);
        glm::mat4 model = glm::mat4(1.0f);
        shader.setMat4("model", model);

        // Bind the VAO
        glBindVertexArray(VAO);

        // Draw the mesh
        glDrawElements(GL_TRIANGLES, faces.size(), GL_UNSIGNED_INT, 0);

        // Unbind the VAO
        glBindVertexArray(0);

        // Swap buffers
        glfwSwapBuffers(window);
    }

    // Clean up
    glDeleteVertexArrays(1, &VAO);
    glDeleteBuffers(1, &VBO);
    glDeleteBuffers(1, &EBO);
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
