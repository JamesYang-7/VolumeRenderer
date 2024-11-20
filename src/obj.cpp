#include <iostream>
#include <fstream>
#include <sstream>
#include "vol_renderer/obj.h"

// Function to read a mesh from an OBJ file
bool readOBJ(const std::string& filename, std::vector<glm::vec3>& vertices, std::vector<GLuint>& faces) {
    std::ifstream infile(filename);
    if (!infile) {
        std::cerr << "Cannot open the file: " << filename << std::endl;
        return false;
    }

    std::string line;
    while (std::getline(infile, line)) {
        std::stringstream ss(line);
        std::string prefix;
        ss >> prefix;
        if (prefix == "v") {
            glm::vec3 v;
            ss >> v.x >> v.y >> v.z;
            vertices.push_back(v);
        } else if (prefix == "f") {
            GLuint a, b, c;
            ss >> a >> b >> c;
            faces.push_back(a - 1);
            faces.push_back(b - 1);
            faces.push_back(c - 1);
        }
    }
    infile.close();
    return true;
}

// Function to write a mesh to an OBJ file
bool writeOBJ(const std::string& filename, const std::vector<glm::vec3>& vertices, const std::vector<GLuint>& faces) {
    std::ofstream outfile(filename);
    if (!outfile) {
        std::cerr << "Cannot open the file for writing: " << filename << std::endl;
        return false;
    }

    for (const auto& v : vertices) {
        outfile << "v " << v.x << " " << v.y << " " << v.z << std::endl;
    }
    for (size_t i = 0; i < faces.size(); i += 3) {
        outfile << "f " << faces[i] + 1 << " " << faces[i + 1] + 1 << " " << faces[i + 2] + 1 << std::endl;
    }
    outfile.close();
    return true;
}