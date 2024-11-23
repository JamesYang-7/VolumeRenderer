#pragma once

#include <GL/glew.h>
#include <GLFW/glfw3.h>

struct Timer {
    float currentFrame = 0.0f;	// time between current frame and last frame
    float lastFrame = 0.0f;

    float getDeltaTime() {
        currentFrame = static_cast<float>(glfwGetTime());
        float deltaTime = currentFrame - lastFrame;
        return deltaTime;
    }

    void update() {
        lastFrame = currentFrame;
    }
};