add_executable(PRangeLiteDemo main.cpp)
target_link_libraries (PRangeLiteDemo libprange)
include_directories("${CMAKE_CURRENT_BINARY_DIR}/../include")
