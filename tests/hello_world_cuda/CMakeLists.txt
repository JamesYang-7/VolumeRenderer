set(APP_NAME hello_world_cuda)
add_executable(${APP_NAME} hello_world_cuda.cu)
target_include_directories(${APP_NAME} PRIVATE ${VOLRENDERER_INCLUDE_DIRS})
target_link_libraries(${APP_NAME} PRIVATE ${VOLRENDERER_LINK_LIBRARIES})