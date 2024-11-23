#pragma once

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include "vol_renderer/camera_gl.h"
#include "vol_renderer/timer.h"


void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void mouse_callback(GLFWwindow* window, double xpos, double ypos);
void scroll_callback(GLFWwindow* window, double xoffset, double yoffset);

struct GUI {
    GLFWwindow* window;
    Timer timer;
    GLCamera* camera;

    GUI(GLFWwindow* window);
    void init();
    void process_input(float deltaTime);
    void process_input();
};
