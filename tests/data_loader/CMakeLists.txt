set(APP_NAME data_loader_test)
add_executable(${APP_NAME}
    data_loader_test.cpp
    ${CMAKE_SOURCE_DIR}/src/data_loader.cpp
)
target_include_directories(${APP_NAME} PRIVATE ${VOLRENDERER_INCLUDE_DIRS})
target_link_libraries(${APP_NAME} PRIVATE ${VOLRENDERER_LINK_LIBRARIES})