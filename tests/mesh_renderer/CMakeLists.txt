set(APP_NAME mesh_renderer)
add_executable(${APP_NAME}
    mesh_renderer.cpp
    ${CMAKE_SOURCE_DIR}/src/gui.cpp
    ${CMAKE_SOURCE_DIR}/src/obj.cpp
)
target_include_directories(${APP_NAME} PRIVATE ${VOLRENDERER_INCLUDE_DIRS})
target_link_libraries(${APP_NAME} PRIVATE ${VOLRENDERER_LINK_LIBRARIES})

file(GLOB SHADERS
    "${CMAKE_SOURCE_DIR}/shaders/mesh/*"
)
message(STATUS "Copying ${APP_NAME} shaders to build directory")
add_custom_command(
    TARGET ${APP_NAME} PRE_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_FILE_DIR:${APP_NAME}>/shaders
    COMMAND ${CMAKE_COMMAND} -E copy ${SHADERS} $<TARGET_FILE_DIR:${APP_NAME}>/shaders
    COMMAND_EXPAND_LISTS
)

file(GLOB DATA "${CMAKE_CURRENT_SOURCE_DIR}/data/*")
message(STATUS "Copying ${APP_NAME} data to build directory")
add_custom_command(
    TARGET ${APP_NAME} PRE_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_FILE_DIR:${APP_NAME}>/data
    COMMAND ${CMAKE_COMMAND} -E copy ${DATA} $<TARGET_FILE_DIR:${APP_NAME}>/data
    COMMAND_EXPAND_LISTS
)