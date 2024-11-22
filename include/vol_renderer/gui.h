#pragma once

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include "vol_renderer/camera_gl.h"

// Window dimensions
extern GLCamera main_camera;

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
