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
On Windows, this should be done in a Developer Command Prompt for VS
```
cmake -B build -G Ninja
cmake --build build
```

#### VSCode
- Install "C/C++ Extension Pack" from VSCode.
- Install & select a compiler (Command Palette -> CMake: Select a Kit)
- Build the target (click "Build" at the bottom or Command Palette -> CMake: Build)
- Launch the program (the play button at the bottom)
