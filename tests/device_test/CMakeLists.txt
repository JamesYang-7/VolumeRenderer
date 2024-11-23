set(APP_NAME device_test)
add_executable(${APP_NAME} device_test.cpp)
target_include_directories(${APP_NAME} PRIVATE ${VOLRENDERER_INCLUDE_DIRS})
target_link_libraries(${APP_NAME} PRIVATE ${VOLRENDERER_LINK_LIBRARIES})