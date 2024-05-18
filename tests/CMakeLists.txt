# Create the test files and CMakeLists.txt files
execute_process(COMMAND ${Python3_EXECUTABLE} create_test_files.py
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

add_subdirectory("compile_time_checks")
add_subdirectory("correctness")