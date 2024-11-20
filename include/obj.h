#pragma once

#include <string>
#include <vector>
#include <GL/glew.h>
#include <glm/glm.hpp>

bool readOBJ(const std::string& filename, std::vector<glm::vec3>& vertices, std::vector<GLuint>& faces);

bool writeOBJ(const std::string& filename, const std::vector<glm::vec3>& vertices, const std::vector<GLuint>& faces);
