set(APP_NAME volume_data_test)
add_executable(${APP_NAME}
    volume_data_test.cpp
    ${CMAKE_SOURCE_DIR}/src/volume_data.cpp
)
target_include_directories(${APP_NAME} PRIVATE ${VOLRENDERER_INCLUDE_DIRS})
target_link_libraries(${APP_NAME} PRIVATE ${VOLRENDERER_LINK_LIBRARIES})