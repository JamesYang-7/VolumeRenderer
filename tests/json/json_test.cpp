#include <json/json.hpp>
#include <fstream>
#include <iostream>
#include <string>

using json = nlohmann::json;

int main() {
    std::ifstream f("config.json");
    json args = json::parse(f);
    std::cout << args["pi"] << ", Type of pi: " << args["pi"].type_name() << std::endl;
    std::cout << args["happy"] << ", Type of happy: " << args["happy"].type_name() << std::endl;
    bool is_happy = args["happy"].get<bool>();
    std::cout << args["name"] << ", Type of name: " << args["name"].type_name() << std::endl;
    std::cout << args["answer"]["everything"] << ", Type of everything: " << args["answer"]["everything"] .type_name() << std::endl;
    std::cout << args["list"] << ", Type of list: " << args["list"].type_name() << std::endl;
    for (const auto& item : args["list"]) {
        std::cout << item << ", Type of item: " << item.type_name() << std::endl;
    }
}