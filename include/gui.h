#pragma once

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include "camera.h"

// Window dimensions
const GLuint WIDTH = 800, HEIGHT = 600;
extern Camera main_camera;

void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void mouse_callback(GLFWwindow* window, double xpos, double ypos);
void scroll_callback(GLFWwindow* window, double xoffset, double yoffset);

struct Timer {
    float currentFrame = 0.0f;	// time between current frame and last frame
    float lastFrame = 0.0f;

    float getDeltaTime();
    void update();
};

struct GUI {
    GLFWwindow* window;

    GUI(GLFWwindow* window);
    void init();
    void process_input(float deltaTime=0);
};
