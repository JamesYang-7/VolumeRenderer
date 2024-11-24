set(APP_NAME json_test)
add_executable(${APP_NAME}
    json_test.cpp
)
target_include_directories(${APP_NAME} PRIVATE ${VOLRENDERER_INCLUDE_DIRS})
target_link_libraries(${APP_NAME} PRIVATE ${VOLRENDERER_LINK_LIBRARIES})

file(GLOB DATA "${CMAKE_CURRENT_SOURCE_DIR}/*.json")
message(STATUS "Copying ${APP_NAME} data to build directory")
add_custom_command(
    TARGET ${APP_NAME} PRE_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy ${DATA} $<TARGET_FILE_DIR:${APP_NAME}>
    COMMAND_EXPAND_LISTS
)