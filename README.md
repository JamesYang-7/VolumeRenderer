## Volume Renderer

### Usage

- Clone the repository
```
git clone --recursive https://github.com/JamesYang-7/VolumeRenderer.git
```

#### Requirements
- CMake 3.23 or higher
- Ninja

#### Command Line
On Windows, this should be done in a Developer Command Prompt. First install VS2022
and search for "x64 native tools command prompt" in the start menu.
```
cmake -B build -G Ninja
cmake --build build
cd build
vol_renderer.exe
```

#### VSCode
- Install "C/C++ Extension Pack" from VSCode.
- Install & select a compiler (Command Palette -> CMake: Select a Kit)
- Build the target (click "Build" at the bottom or Command Palette -> CMake: Build)
- Launch the program (the play button at the bottom)
